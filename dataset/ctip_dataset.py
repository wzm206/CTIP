import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb
import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import jpeg4py as jpeg
import cv2

from PIL import Image
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
# test code
import sys
sys.path.append(os.getcwd())
print(sys.path)

from dataset.data_utils import (
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class Dataset_CTIP(Dataset):
    def __init__(
        self,
        config,
        train_or_test:str,
        dataset_name,
    ):
        self.config = config
        self.down_rate = config[dataset_name]["down_rate"]
        self.data_folder_posi = os.path.join(config[dataset_name]["output_dir"], "data")
        traj_names_file_posi = os.path.join(config[dataset_name]["output_dir"], train_or_test+"_traj_names.txt")

        with open(traj_names_file_posi, "r") as f:
            file_lines = f.read()
            self.traj_names_posi = file_lines.split("\n")
        if "" in self.traj_names_posi:
            self.traj_names_posi.remove("")

        self.image_size = config["image_size"]
        self.len_traj_pred = config["len_traj_pred"]
        self.context_size = config["context_size"]
        # 算出多少个间隔采一次样
        self.waypoint_spacing = config[dataset_name]["waypoint_spacing"]

        # use this index to retrieve the dataset name from the data_config.yaml
        self.trajectory_cache = {}
        self.index_data_dic = self._build_index()
        

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        index_data_dic = {}
        index_data_dic["posi"] = []
        index_data_dic["nega"] = []

        print("process posi data")
        for traj_name in tqdm.tqdm(self.traj_names_posi, disable=not use_tqdm, dynamic_ncols=True):
            with open(os.path.join(self.data_folder_posi, traj_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            # format:
            # traj_data_dic = {"traj_name":traj_name_i, "traj_time":time_data_i, "traj_data":traj_data_i}
            # traj_data_i = {"position": np.array(xys), "yaw": np.array(yaws)}
            traj_len = len(traj_data["traj_data"]["position"])
            begin_index = self.context_size * self.waypoint_spacing
            end_index = traj_len - self.len_traj_pred * self.waypoint_spacing
   
            for curr_index in range(begin_index, end_index, self.down_rate):
                sigle_data_dic = {}
                img_index = range(curr_index, curr_index-self.context_size*self.waypoint_spacing, -self.waypoint_spacing)
                sigle_data_dic["img_index"] = img_index
                traj_index = range(curr_index, curr_index+self.len_traj_pred*self.waypoint_spacing, self.waypoint_spacing)
                sigle_data_dic["traj_index"] = traj_index
                now_position, now_yaw = traj_data["traj_data"]["position"][curr_index], traj_data["traj_data"]["yaw"][curr_index]
                abs_position_pre = traj_data["traj_data"]["position"][traj_index]
                sigle_data_dic["position_predict_data"] = to_local_coords(abs_position_pre, now_position, now_yaw)
                sigle_data_dic["time_predict_data"] = traj_data["traj_time"][traj_index]
                abs_position_img =traj_data["traj_data"]["position"][img_index]
                sigle_data_dic["position_img_data"] = to_local_coords(abs_position_img, now_position, now_yaw)
                sigle_data_dic["time_img_data"] = traj_data["traj_time"][img_index]
                sigle_data_dic["traj_name"] = traj_data["traj_name"]
                index_data_dic["posi"].append(sigle_data_dic)
                
                
        print("data has {} items".format(len(index_data_dic["posi"])) )
        return index_data_dic





    def __len__(self) -> int:
        return max(len(self.index_data_dic["posi"]),len(self.index_data_dic["nega"]))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:

        posi_index = i
        posi_data = self.index_data_dic["posi"][posi_index]

        
        obs_images_posi_list =[]
        # img_index from now to t-1 t-2 ...
        for now_img_index in posi_data["img_index"]:
            img_path = os.path.join(self.data_folder_posi, posi_data["traj_name"], str(now_img_index)+".jpg")
            # img = jpeg.JPEG(img_path).decode()
            now_img = Image.open(img_path).resize(self.image_size)
            obs_images_posi_list.append(transform(now_img))
        torch.stack(obs_images_posi_list, 0)
            

        return (
            torch.stack(obs_images_posi_list, 0),
            torch.as_tensor(posi_data["position_predict_data"], dtype=torch.float32),
            torch.as_tensor(posi_data["position_img_data"], dtype=torch.float32),
        )




def get_CTIP_loader(config):
    dataset_name = "bionic"
    dataset_train = Dataset_CTIP(
            config=config,
            train_or_test="train",
            dataset_name=dataset_name,
        )
    dataset_test = Dataset_CTIP(
            config=config,
            train_or_test="test",
            dataset_name=dataset_name,
        )


    loader_train = DataLoader(
        dataset_train,
        pin_memory=True,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    loader_test = DataLoader(
        dataset_test,
        pin_memory=True,
        batch_size=config["test_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    return loader_train, loader_test







# test code

# with open("config/ctip.yaml", "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
    
# dataset = Dataset_CTIP(
#         config=config,
#         train_or_test="train",
#         dataset_name="bionic",
#     )
# from torch.utils.data import DataLoader, ConcatDataset

# untransform = transforms.Compose([
#     transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
#                          std=[1/0.229, 1/0.224, 1/0.225]),
# ])

# loader = DataLoader(
#     dataset,
#     batch_size=66,
#     shuffle=True,
#     num_workers=0,
#     drop_last=False,
# )
# import matplotlib.pyplot as plt
# for data in loader:
#     (obs_images_posi,
#     waypoint_posi,
#     img_position_posi) = data
    
#     now_img = obs_images_posi[0][0]
#     img_np = untransform(now_img).cpu().detach().numpy()
#     f, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.imshow(img_np.transpose(1,2,0))
#     label_np = waypoint_posi[0].cpu().detach().numpy()
#     ax2.plot(-label_np[:,1],label_np[:,0], c="b")
#     plt.xlim(-10, 10)  # 设定绘图范围
#     plt.ylim(0, 20) 
#     plt.show()
    