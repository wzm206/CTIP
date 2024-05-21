import numpy as np
import io
import os
import rosbag
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import pickle
import torchvision.transforms.functional as TF


IMAGE_SIZE = (224, 224)
IMAGE_ASPECT_RATIO = 4 / 3


def process_images(im_list: List, img_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    images = []
    for img_msg in im_list:
        img = img_process_func(img_msg)
        images.append(img)
    return images


def process_tartan_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the tartan_drive dataset
    """
    img = ros_to_numpy(msg, output_resolution=IMAGE_SIZE) * 255
    img = img.astype(np.uint8)
    # reverse the axis order to get the image in the right orientation
    img = np.moveaxis(img, 0, -1)
    # convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    return img


def process_locobot_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the locobot dataset
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = Image.fromarray(img)
    return pil_image
# --------------以下自己修改-----------------
def process_wzm_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the locobot dataset
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    img = img[:,:,[2,1,0]]
    pil_image = Image.fromarray(img)
    return pil_image.resize(IMAGE_SIZE)
def process_daheng_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the locobot dataset
    """
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
    # resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE)
    return img
# --------------修改完成-----------------

def process_scand_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the scand dataset
    """
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
    # center crop image to 4:3 aspect ratio
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * IMAGE_ASPECT_RATIO))
    )  # crop to the right ratio
    # resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE)
    return img


############## Add custom image processing functions here #############

def process_sacson_img(msg) -> Image:
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_np)
    return pil_image


#######################################################################


def process_odom(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    yaws = []
    for odom_msg in odom_list:
        xy, yaw = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
    return {"position": np.array(xys), "yaw": np.array(yaws)}


def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """

    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y], yaw


############ Add custom odometry processing functions here ############


#######################################################################


def get_images_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    Get image and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        odomtopics (list[str] or str): topic name(s) for odom data
        img_process_func (Any): function to process image data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
    Returns:
        img_data (list): list of PIL images
        traj_data (list): list of odom data
    """
    # check if bag has both topics
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if not (imtopic and odomtopic):
        # bag doesn't have both topics
        return None, None

    synced_imdata = []
    synced_odomdata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()
    starttime = currtime

    curr_imdata = None
    curr_odomdata = None
    times = []

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
            currtime = t.to_sec()
            times.append(currtime - starttime)

    img_data = process_images(synced_imdata, img_process_func)
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )

    return img_data, traj_data

# *******************以下自己*****************************
def get_nega_images_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    collisiontopics: List[str] or str,
    lanetopics: List[str] or str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,

):
    """
    Get image and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        odomtopics (list[str] or str): topic name(s) for odom data
        img_process_func (Any): function to process image data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
    Returns:
        img_data (list): list of PIL images
        traj_data (list): list of odom data
    """
    # check if bag has both topics
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if not (imtopic and odomtopic):
        # bag doesn't have both topics
        return None, None

    synced_imdata = []
    synced_odomdata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()
    starttime = currtime

    curr_imdata = None
    curr_odomdata = None
    times = []

    ret_img_data = []
    ret_traj_data = []
    cut = False

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic, collisiontopics, lanetopics]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        elif topic == collisiontopics:
            curr_colldata = msg
            cut = True
        elif topic == lanetopics:
            cut = True
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
            currtime = t.to_sec()
            times.append(currtime - starttime)

        # 及时截断，有cut说明有碰撞信号了
        if cut:
            if len(synced_imdata)>5 + 1:
                img_data = process_images(synced_imdata, img_process_func)
                traj_data = process_odom(
                    synced_odomdata,
                    odom_process_func,
                    ang_offset=ang_offset,
                )
                ret_img_data.append(img_data)
                ret_traj_data.append(traj_data)
            cut = False
            curr_imdata = None
            curr_odomdata = None
            synced_imdata.clear()
            synced_odomdata.clear()



    return ret_img_data, ret_traj_data


# *******************以下自己*****************************
def get_posi_images_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    collisiontopics: List[str] or str,
    lanetopics: List[str] or str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,

):
    """
    Get image and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        odomtopics (list[str] or str): topic name(s) for odom data
        img_process_func (Any): function to process image data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
    Returns:
        img_data (list): list of PIL images
        traj_data (list): list of odom data
    """
    # check if bag has both topics
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if not (imtopic and odomtopic):
        # bag doesn't have both topics
        return None, None

    synced_imdata = []
    synced_odomdata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()
    starttime = currtime

    curr_imdata = None
    curr_odomdata = None
    times = []

    ret_img_data = []
    ret_traj_data = []
    cut = False

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic, collisiontopics, lanetopics]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        elif topic == collisiontopics:
            curr_colldata = msg
            cut = True
        elif topic == lanetopics:
            cut = True
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
            currtime = t.to_sec()
            times.append(currtime - starttime)

        # 及时截断，有cut说明有碰撞信号了

        if len(synced_imdata)>99:
        # if cut:
            # if len(synced_imdata) > 8:
            img_data = process_images(synced_imdata, img_process_func)
            traj_data = process_odom(
                synced_odomdata,
                odom_process_func,
                ang_offset=ang_offset,
            )
            ret_img_data.append(img_data)
            ret_traj_data.append(traj_data)
            cut = False
            curr_imdata = None
            curr_odomdata = None
            synced_imdata.clear()
            synced_odomdata.clear()



    return ret_img_data, ret_traj_data


# ---------------------真正自己的------------------------
def get_carla_images_and_odom(
    bag: rosbag.Bag,
    config,
    traj_name,
    output_path,
    ang_offset: float = 0.0,
):
    img_process_func = process_wzm_img
    odom_process_func = nav_to_xy_yaw
    imtopic, odomtopic = config["IMAGE_TOPIC"], config["ODOM_TOPIC"]

    synced_imdata = []
    synced_odomdata = []
    synced_times = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()
    starttime = currtime

    curr_imdata = None
    curr_odomdata = None
    traj_index = 0

    cut = False

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
            if curr_odomdata.twist.twist.linear.x<0.05:
                cut = True

        if curr_imdata is not None and curr_odomdata is not None:
            synced_imdata.append(curr_imdata)
            synced_odomdata.append(curr_odomdata)
            curr_imdata = None
            curr_odomdata = None
            currtime = t.to_sec()
            synced_times.append(currtime - starttime)

        # 及时截断，有cut说明有碰撞信号了

        if len(synced_imdata)>999:
            cut = True
        
        if cut:
            if len(synced_imdata) > 50:
                img_data = process_images(synced_imdata, img_process_func)
                traj_data = process_odom(
                    synced_odomdata,
                    odom_process_func,
                    ang_offset=ang_offset,
                )
                assert len(img_data)==len(synced_times)
                
                # 有很多段小轨迹，这是其中的一段
                img_data_i = img_data
                traj_data_i = traj_data
                time_data_i = np.array(synced_times)
                traj_name_i = traj_name + f"_{traj_index}"
                traj_index += 1
                
                traj_folder_i = os.path.join(output_path, traj_name_i)
                # make a folder for the traj
                if not os.path.exists(traj_folder_i):
                    os.makedirs(traj_folder_i)
                with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                    traj_data_dic = {"traj_name":traj_name_i, "traj_time":time_data_i, "traj_data":traj_data_i}
                    pickle.dump(traj_data_dic, f)
                # save the image data to disk
                for i, img in enumerate(img_data_i):
                    img.save(os.path.join(traj_folder_i, f"{i}.jpg"))
                
                
                
            cut = False
            curr_imdata = None
            curr_odomdata = None
            synced_imdata.clear()
            synced_odomdata.clear()
            synced_times.clear()

    
def get_ctip_images_and_odom(
    bag: rosbag.Bag,
    config,
    traj_name,
    dataset_name,
    ang_offset: float = 0.0,
):
    output_path=os.path.join(config[dataset_name]["output_dir"], "data")
    img_process_func = eval(config[dataset_name]["img_process_func"])
    odom_process_func = eval(config[dataset_name]["odom_process_func"])
    imtopic, odomtopic = config[dataset_name]["IMAGE_TOPIC"], config[dataset_name]["ODOM_TOPIC"]

    synced_imdata = []
    synced_odomdata = []
    synced_times = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()
    starttime = currtime

    curr_imdata = None
    curr_odomdata = None
    traj_index = 0

    cut = False

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
            if curr_odomdata.twist.twist.linear.x<0.05:
                cut = True

        if curr_imdata is not None and curr_odomdata is not None:
            synced_imdata.append(curr_imdata)
            synced_odomdata.append(curr_odomdata)
            curr_imdata = None
            curr_odomdata = None
            currtime = t.to_sec()
            synced_times.append(currtime - starttime)

        # 及时截断，有cut说明有碰撞信号了

        if len(synced_imdata)>999:
            cut = True
        
        if cut:
            if len(synced_imdata) > 50:
                img_data = process_images(synced_imdata, img_process_func)
                traj_data = process_odom(
                    synced_odomdata,
                    odom_process_func,
                    ang_offset=ang_offset,
                )
                assert len(img_data)==len(synced_times)
                
                # 有很多段小轨迹，这是其中的一段
                img_data_i = img_data
                traj_data_i = traj_data
                time_data_i = np.array(synced_times)
                traj_name_i = traj_name + f"_{traj_index}"
                traj_index += 1
                
                traj_folder_i = os.path.join(output_path, traj_name_i)
                # make a folder for the traj
                if not os.path.exists(traj_folder_i):
                    os.makedirs(traj_folder_i)
                with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                    traj_data_dic = {"traj_name":traj_name_i, "traj_time":time_data_i, "traj_data":traj_data_i}
                    pickle.dump(traj_data_dic, f)
                # save the image data to disk
                for i, img in enumerate(img_data_i):
                    img.save(os.path.join(traj_folder_i, f"{i}.jpg"))
                
                
                
            cut = False
            curr_imdata = None
            curr_odomdata = None
            synced_imdata.clear()
            synced_odomdata.clear()
            synced_times.clear()    


# *******************以上自己*****************************

def get_push_pull_images_and_odom_bike(
    is_posi: bool,
    bag: rosbag.Bag,
    imtopic: str,
    odomtopic: str,
    reborn_topic: str,
    img_type: str = "CompressedImage",
    rate: float = -1,
    ang_offset: float = 0.0,
):

    if img_type=="Image":
        img_process_func = process_wzm_img
    odom_process_func = nav_to_xy_yaw

    synced_imdata = []
    synced_odomdata = []
    synced_times = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()
    starttime = currtime

    curr_imdata = None
    curr_odomdata = None
    

    ret_img_data = []
    ret_traj_data = []
    ret_time_data = []
    cut = False
    time_next_start = 0

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic, reborn_topic]):
        if t.to_sec()<time_next_start:
            continue
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
            if curr_odomdata.twist.twist.linear.x<0.05:
                cut = True
        if (t.to_sec() - currtime) >= 1.0 / rate or rate==-1:
            # should always go into here
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                curr_imdata = None
                curr_odomdata = None
                currtime = t.to_sec()
                synced_times.append(currtime - starttime)

        # 及时截断，有cut说明有碰撞信号了

        if len(synced_imdata)>999 and is_posi:
            cut = True
        
        if cut:
            if len(synced_imdata) > 5:
                img_data = process_images(synced_imdata, img_process_func)
                traj_data = process_odom(
                    synced_odomdata,
                    odom_process_func,
                    ang_offset=ang_offset,
                )
                assert len(img_data)==len(synced_times)
                ret_img_data.append(img_data)
                ret_traj_data.append(traj_data)
                ret_time_data.append(np.array(synced_times))
            cut = False
            curr_imdata = None
            curr_odomdata = None
            synced_imdata.clear()
            synced_odomdata.clear()
            synced_times.clear()
            time_next_start = t.to_sec()+0.2

    return ret_img_data, ret_traj_data, ret_time_data


# *******************以上自己*****************************

def is_backwards(
    pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5
) -> bool:
    """
    Check if the trajectory is going backwards given the position and yaw of two points
    Args:
        pos1: position of the first point

    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps


# cut out non-positive velocity segments of the trajectory
def filter_backwards(
    img_list: List[Image.Image],
    traj_data: Dict[str, np.ndarray],
    start_slack: int = 0,
    end_slack: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Cut out non-positive velocity segments of the trajectory
    Args:
        traj_type: type of trajectory to cut
        img_list: list of images
        traj_data: dictionary of position and yaw data
        start_slack: number of points to ignore at the start of the trajectory
        end_slack: number of points to ignore at the end of the trajectory
    Returns:
        cut_trajs: list of cut trajectories
        start_times: list of start times of the cut trajectories
    """
    traj_pos = traj_data["position"]
    traj_yaws = traj_data["yaw"]
    cut_trajs = []
    start = True

    def process_pair(traj_pair: list) -> Tuple[List, Dict]:
        new_img_list, new_traj_data = zip(*traj_pair)
        new_traj_data = np.array(new_traj_data)
        new_traj_pos = new_traj_data[:, :2]
        new_traj_yaws = new_traj_data[:, 2]
        return (new_img_list, {"position": new_traj_pos, "yaw": new_traj_yaws})

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        if not is_backwards(pos1, yaw1, pos2):
            if start:
                new_traj_pairs = [
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                ]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append(
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                )
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))
            start = True
    return cut_trajs


def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def ros_to_numpy(
    msg, nchannels=3, empty_value=None, output_resolution=None, aggregate="none"
):
    """
    Convert a ROS image message to a numpy array
    """
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    is_rgb = "8" in msg.encoding
    if is_rgb:
        data = np.frombuffer(msg.data, dtype=np.uint8).copy()
    else:
        data = np.frombuffer(msg.data, dtype=np.float32).copy()

    data = data.reshape(msg.height, msg.width, nchannels)

    if empty_value:
        mask = np.isclose(abs(data), empty_value)
        fill_value = np.percentile(data[~mask], 99)
        data[mask] = fill_value

    data = cv2.resize(
        data,
        dsize=(output_resolution[0], output_resolution[1]),
        interpolation=cv2.INTER_AREA,
    )

    if aggregate == "littleendian":
        data = sum([data[:, :, i] * (256**i) for i in range(nchannels)])
    elif aggregate == "bigendian":
        data = sum([data[:, :, -(i + 1)] * (256**i) for i in range(nchannels)])

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    else:
        data = np.moveaxis(data, 2, 0)  # Switch to channels-first

    if is_rgb:
        data = data.astype(np.float32) / (
            255.0 if aggregate == "none" else 255.0**nchannels
        )

    return data


def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
