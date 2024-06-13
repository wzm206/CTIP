import numpy as np
import torch
import torch.nn.functional as F
import yaml
import time,datetime
import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from model.CTIP import CTIPModel
from dataset.ctip_dataset import get_CTIP_loader_from_list 
from utils import *
untransform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                         std=[1/0.229, 1/0.224, 1/0.225]),
])


def main(args):
    
    with open("config/ctip.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ret_traj = {}
    train_loader, test_loader = get_CTIP_loader_from_list(config, ["carla"])
    model = CTIPModel().to(config["device"])
    model = load_model_para(model, config)
    model.eval()
    traj_dic = torch.load("./sample_traj_ctip_carla.pt")
    waypoint_ori_all = traj_dic["waypoint_ori_train"].to(config["device"])
    waypoint_normal_train = traj_dic["waypoint_normal_train"].to(config["device"])
    
    loader_tqdm_test = tqdm.tqdm(test_loader, desc="Test Batch", leave=False)
    with torch.no_grad():
        loss = 0
        for i, (obs_images, waypoint, img_position) in enumerate(loader_tqdm_test):
            waypoint=waypoint.to(config["device"])
            batch_data = {}
            
            # waypoint_normal_train = waypoint_normalize(waypoint, config).transpose(1, 2).to(config["device"])


            tem = obs_images[0].to(config["device"]).squeeze()
            
            batch_score = model.get_score_deploy(tem, waypoint_normal_train)
            _, top5_index = torch.topk(batch_score, k=5, dim=0, largest=True, sorted=True)  # k=2
            _, last5_index = torch.topk(batch_score, k=5, dim=0, largest=False, sorted=True)  # k=2
            top5_data = to_numpy(torch.index_select(waypoint_ori_all, dim=0, index=top5_index))
            last5_data = to_numpy(torch.index_select(waypoint_ori_all, dim=0, index=last5_index))
            
            gt_data = to_numpy(waypoint[0])
            # visualize
            f, (all_ax) = plt.subplots(1, 2)
            image_now_numpy = to_numpy(untransform(tem))
            print(image_now_numpy.shape)
            all_ax[0].imshow(image_now_numpy.transpose(1,2,0))
            # gt
            all_ax[1].plot(-gt_data[:,1],gt_data[:,0], c="g", alpha=0.8, linewidth = 5)
            for i_waypoint_index, waypoint_now in enumerate(top5_data):
                all_ax[1].plot(-waypoint_now[:,1],waypoint_now[:,0], c="b", alpha=0.3)
            for i_waypoint_index, waypoint_now in enumerate(last5_data):
                all_ax[1].plot(-waypoint_now[:,1],waypoint_now[:,0], c="r", alpha=0.3)
            
            all_ax[1].set_xlim([config["y_min"]-3, config["y_max"]+3])
            all_ax[1].set_ylim([config["x_min"], config["x_max"]+5])
            plt.show()
            
        loss /= len(test_loader)

    

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')


    args = parser.parse_args()


    
    args.steps = 0

    main(args)
