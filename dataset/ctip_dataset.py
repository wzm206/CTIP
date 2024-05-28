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

horizon = transforms.RandomHorizontalFlip(p=1)
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
        with open(f'./data/{dataset_name}/{train_or_test}_traj_names.pkl', 'rb') as f:
            # 读取并反序列化数据
            data_loaded = pickle.load(f)
        self.data_list = data_loaded
        self.data_folder = os.path.join("./data", dataset_name, "data")
        print(f"{dataset_name}/{train_or_test} has {len(data_loaded)} data")



    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:

        posi_index = i
        posi_data = self.data_list[posi_index]
        inverse = False
        if torch.rand(1) < 0.5:
            inverse = True
        
        obs_images_posi_list =[]
        # img_index from now to t-1 t-2 ...
        for now_img_index in posi_data["img_index"]:
            img_path = os.path.join(self.data_folder, posi_data["traj_name"], str(now_img_index)+".jpg")
            # img = jpeg.JPEG(img_path).decode()
            # now_img = Image.open(img_path).resize(self.image_size)
            # to do 
            now_img = Image.open(img_path)
            if inverse:
                now_img = horizon(transform(now_img))
            else:
                now_img = transform(now_img)
            obs_images_posi_list.append(now_img)
        torch.stack(obs_images_posi_list, 0)
        
        
        #process traj
        traj_data = posi_data["position_predict_data"]
        if inverse:
            traj_data[:, 1] =  -traj_data[:, 1]

        return (
            torch.stack(obs_images_posi_list, 0),
            torch.as_tensor(traj_data, dtype=torch.float32),
            torch.as_tensor(posi_data["position_img_data"], dtype=torch.float32),
        )


def get_casia_loader(config):
    dataset_name_list = ["casia_rgb"]
    dataset_train_list, dataset_test_list = [], []
    for dataset_name in dataset_name_list:
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
        dataset_train_list.append(dataset_train)
        dataset_test_list.append(dataset_test)

    dataset_train, dataset_test = ConcatDataset(dataset_train_list), ConcatDataset(dataset_test_list)
    
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

def get_CTIP_loader(config):
    dataset_name_list = ["bionic", "zju", "rgb_loop"]
    # dataset_name_list = ["sacson_test"]
    dataset_train_list, dataset_test_list = [], []
    for dataset_name in dataset_name_list:
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
        dataset_train_list.append(dataset_train)
        dataset_test_list.append(dataset_test)

    dataset_train, dataset_test = ConcatDataset(dataset_train_list), ConcatDataset(dataset_test_list)
    
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

def get_CTIP_indoor_loader(config):
    # dataset_name_list = ["bionic", "zju", "rgb_loop"]
    dataset_name_list = ["SACSoN"]
    dataset_train_list, dataset_test_list = [], []
    for dataset_name in dataset_name_list:
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
        dataset_train_list.append(dataset_train)
        dataset_test_list.append(dataset_test)

    dataset_train, dataset_test = ConcatDataset(dataset_train_list), ConcatDataset(dataset_test_list)
    
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





# # test code

# with open("config/ctip.yaml", "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# from torch.utils.data import DataLoader, ConcatDataset
# untransform = transforms.Compose([
#     transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
#                          std=[1/0.229, 1/0.224, 1/0.225]),
# ])
# train_loader, test_loader = get_CTIP_loader(config)
# import matplotlib.pyplot as plt
# for data in train_loader:
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
#     plt.ylim(-1, 10) 
#     plt.show()
    