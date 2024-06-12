import cv2
import torch
import yaml
from torchvision import transforms
from model.CTIP import CTIPModel
from utils import *

device = "cuda"
with open("config/ctip.yaml", "r") as f:
    config = yaml.safe_load(f)
video_path = "/home/wzm/data/CTIP_data/cell_phone/road2.mp4"
model = CTIPModel().to(config["device"])
model = load_model_para(model, config)
model.eval()



cap = cv2.VideoCapture(video_path)     # 读取视频
to_tensor = transforms.ToTensor()

batch_data = {}
traj_dic = torch.load("./sample_traj_ctip_casia.pt")
waypoint = traj_dic["waypoint_ori_train"].to(device)
waypoint_normal_train = traj_dic["waypoint_normal_train"].to(device)
batch_data["traj"] = waypoint_normal_train
batch_size = config["test_batch_size"]
# obs_images = transform_images(context_queue, config["image_size"], center_crop=False)[0]




while cap.isOpened():               # 当视频被打开时：
    ret, frame = cap.read()         # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
    if ret:
        # 720, 1280, 3
        h, w, c = frame.shape
        # 600*450
        new_frame = frame[:450, w//2-300:w//2+300]
        
        image_now = cv2.resize(new_frame, (224, 224))
        obs_images = to_tensor(image_now)
        chanel, w, h = obs_images.shape
        same_imgs = obs_images.expand(batch_size, chanel, w, h ).to(device)
        batch_data["image"] = same_imgs
        batch_score = model.get_score(batch_data)[:, 0]
        # print(batch_score)
        _, top5_index = torch.topk(batch_score, k=5, dim=0, largest=True, sorted=True)  # k=2
        _, last5_index = torch.topk(batch_score, k=5, dim=0, largest=False, sorted=True)  # k=2
        # print(top5_index[0:3])
        top5_data = to_numpy(torch.index_select(waypoint, dim=0, index=top5_index))
        last5_data = to_numpy(torch.index_select(waypoint, dim=0, index=last5_index))
        background = np.full((720, 500, 3), 255, dtype=np.uint8)
        
        waypoints_int = (50*top5_data).astype(np.int)
        waypoints_int = waypoints_int[:, :, [1,0]]
        waypoints_int[:, :, 0] = -1*waypoints_int[:, :, 0] + 500//2
        waypoints_int[:, :, 1] = 720-waypoints_int[:, :, 1]
        cv2.polylines(background, waypoints_int, isClosed=False, color=(255,0,0), thickness=2)
        waypoints_int = (50*last5_data).astype(np.int)
        waypoints_int = waypoints_int[:, :, [1,0]]
        waypoints_int[:, :, 0] = -1*waypoints_int[:, :, 0] + 500//2
        waypoints_int[:, :, 1] = 720-waypoints_int[:, :, 1]
        cv2.polylines(background, waypoints_int, isClosed=False, color=(0,0,255), thickness=1)
        
        full_image = np.concatenate([frame, background], axis=1)
        cv2.imshow('frame', full_image)  # 显示读取到的这一帧画面
        # cv2.imshow('frame', full_image)  # 显示读取到的这一帧画面
        
        key = cv2.waitKey(0)       # 等待一段时间，并且检测键盘输入
        if key == ord('q'):         # 若是键盘输入'q',则退出，释放视频
            cap.release()           # 释放视频
            break
    else:
        cap.release()
cv2.destroyAllWindows()             # 关闭所有窗口
