import numpy as np
import torch
import torch.nn.functional as F
import yaml
import time,datetime

from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from model.CTIP import CTIPModel
from dataset.ctip_dataset import get_CTIP_loader_from_list 
from utils import waypoint_normalize, waypoint_unnormalize


def main(args):
    
    with open("config/ctip.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ret_traj = {}
    train_loader, test_loader = get_CTIP_loader_from_list(config, ["rgb_loop"])
    obs_images, waypoint, img_position = next(iter(train_loader))
    ret_traj["waypoint_ori_train"] = waypoint
    ret_traj["waypoint_normal_train"] = waypoint_normalize(waypoint.to(config["device"]), 
                                            config).transpose(1, 2)
    
    obs_images, waypoint, img_position = next(iter(train_loader))
    ret_traj["waypoint_ori_test"] = waypoint
    ret_traj["waypoint_normal_test"] = waypoint_normalize(waypoint.to(config["device"]), 
                                            config).transpose(1, 2)
    torch.save(ret_traj, "./sample_traj_ctip_casia.pt")

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')


    args = parser.parse_args()


    
    args.steps = 0

    main(args)
