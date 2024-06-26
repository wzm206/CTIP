import matplotlib.pyplot as plt
import os
import sys
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt
import yaml

# ROS
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray

# from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
sys.path.insert(0,sys.path[0]+"/../")
from model.CTIP import CTIPModel
from utils import *
from simple_pid import PID


pid = PID(0.3, 0, 0, setpoint=0, output_limits=(-0.4, 0.4))

# GLOBALS
context_queue = []
context_size = None  

# Load the model 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def callback_obs(msg):
    obs_img = com_msg_to_pil(msg)
    if context_size is not None:
        # 注意先后顺序
        if len(context_queue) < context_size:
            context_queue.insert(0, obs_img)
        else:
            context_queue.pop(-1)
            context_queue.insert(0, obs_img)


def main(config):
    global context_size

    context_size = config["context_size"]
    config["ckpt_path"] = config[args.deploy_env]["ckpt_path"]
    print("load traj from:", config["ckpt_path"])
    model = CTIPModel().to(config["device"])
    model = load_model_para(model, config)
    model.eval()

    # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(config["ros_rate"])
    image_curr_msg = rospy.Subscriber(
        config["rgb_loop"]["IMAGE_TOPIC"], CompressedImage, callback_obs, queue_size=1)
    carla_twist_pub = rospy.Publisher(config["CONTROL_TOPIC"], Twist, queue_size=1)

    keshihua_pub1_posi = rospy.Publisher(config["posi_waypoints_topic"] + "1", Path, queue_size=10)
    keshihua_pub2_posi = rospy.Publisher(config["posi_waypoints_topic"] + "2", Path, queue_size=10)
    keshihua_pub3_posi = rospy.Publisher(config["posi_waypoints_topic"] + "3", Path, queue_size=10)
    keshihua_pub4_posi = rospy.Publisher(config["posi_waypoints_topic"] + "4", Path, queue_size=10)
    keshihua_pub5_posi = rospy.Publisher(config["posi_waypoints_topic"] + "5", Path, queue_size=10)

    
    keshihua_pub1_nega = rospy.Publisher(config["nega_waypoints_topic"] + "1", Path, queue_size=10)
    keshihua_pub2_nega = rospy.Publisher(config["nega_waypoints_topic"] + "2", Path, queue_size=10)
    keshihua_pub3_nega = rospy.Publisher(config["nega_waypoints_topic"] + "3", Path, queue_size=10)
    keshihua_pub4_nega = rospy.Publisher(config["nega_waypoints_topic"] + "4", Path, queue_size=10)
    keshihua_pub5_nega = rospy.Publisher(config["nega_waypoints_topic"] + "5", Path, queue_size=10)

    batch_data = {}
    traj_dic = torch.load(config[args.deploy_env]["traj_path"])
    print("load traj from:", config[args.deploy_env]["traj_path"])
    
    waypoint_ori_train = traj_dic["waypoint_ori_train"].to(device)
    waypoint_normal_train = traj_dic["waypoint_normal_train"].to(device)
    batch_data["traj"] = waypoint_normal_train
        
    print("Registered with master node. Waiting for image observations...")


    while not rospy.is_shutdown():
        # EXPLORATION MODE
        if (
                len(context_queue) == context_size
            ):

            obs_images = transform_images(context_queue, config["image_size"], center_crop=False)[0].to(device)
            # chanel, w, h = obs_images.shape
            batch_data["image"] = obs_images
            batch_score = model.get_score_deploy(obs_images, waypoint_normal_train)
            _, top5_index = torch.topk(batch_score, k=5, dim=0, largest=True, sorted=True)  # k=2
            _, last5_index = torch.topk(batch_score, k=5, dim=0, largest=False, sorted=True)  # k=2
            top5_data = to_numpy(torch.index_select(waypoint_ori_train, dim=0, index=top5_index))
            last5_data = to_numpy(torch.index_select(waypoint_ori_train, dim=0, index=last5_index))

            
            action = top5_data[0] # change this based on heuristic
            chosen_waypoint = action[args.waypoint]

            # calculate control input
            vel_msg = Twist()
            v = 0.5
            w = np.arctan(chosen_waypoint[1]/chosen_waypoint[0])
            pid_out = pid(-w)
            vel_msg.linear.x = v
            vel_msg.angular.z = pid_out
            
            print(f"publishing new vel: {v}, {w}")
            
            carla_twist_pub.publish(vel_msg)

            # 发布可视化信息
            keshihua_pub_posi_list = [keshihua_pub1_posi, keshihua_pub2_posi, keshihua_pub3_posi, keshihua_pub4_posi,
                                      keshihua_pub5_posi]
            keshihua_pub_nega_list = [keshihua_pub1_nega, keshihua_pub2_nega, keshihua_pub3_nega, keshihua_pub4_nega,
                                      keshihua_pub5_nega]
            
            keshihua_pub_list = [keshihua_pub_posi_list, keshihua_pub_nega_list]
            waypoints_list = [top5_data, last5_data]
            
            for i_loop in range(2):
                keshihua_list = keshihua_pub_list[i_loop]
                waypoints_to_pub = waypoints_list[i_loop]
                for k in range(5):
                    keshihua_msg = Path()
                    keshihua_msg.header.frame_id = "ego_vehicle"
                    keshihua_msg.header.stamp =  rospy.Time.now()
                    for now in waypoints_to_pub[k]:
                        keshihua_pose = PoseStamped()
                        # print("datais:", now, "typeis", type(now))
                        keshihua_pose.header = keshihua_msg.header
                        # keshihua_pose.header.stamp =  rospy.Time.now()
                        keshihua_pose.pose.position.x = now[0]
                        keshihua_pose.pose.position.y = now[1]
                        keshihua_pose.pose.position.z = 0
                        keshihua_msg.poses.append(keshihua_pose)
                    keshihua_list[k].publish(keshihua_msg)


        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")

    parser.add_argument(
        "--waypoint",
        "-w",
        default=6, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    
    parser.add_argument(
        "--config",
        "-c",
        default="config/ctip.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    
    parser.add_argument(
        "--deploy_env",
        "-de",
        default="data_casia",
        type=str,
        help="choose which model",
    )
    
    args = parser.parse_args()


    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Using {device}")
    main(config)


