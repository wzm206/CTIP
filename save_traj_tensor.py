import numpy as np
import torch
import torch.nn.functional as F
import yaml
import time,datetime

from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from model.CTIP import CTIPModel
from dataset.carla import get_Carla_loader
from utils import waypoint_normalize, waypoint_unnormalize


def main(args):
    
    with open("config/carla.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ret_traj = {}
    train_loader, test_loader = get_Carla_loader(config)
    obs_images, waypoint, img_position = next(iter(train_loader))
    ret_traj["waypoint_ori_train"] = waypoint
    ret_traj["waypoint_normal_train"] = waypoint_normalize(waypoint.to(config["device"]), 
                                            config["min_x"], config["max_x"],
                                            config["min_y"],config["max_y"]).transpose(1, 2)
    
    obs_images, waypoint, img_position = next(iter(train_loader))
    ret_traj["waypoint_ori_test"] = waypoint
    ret_traj["waypoint_normal_test"] = waypoint_normalize(waypoint.to(config["device"]), 
                                            config["min_x"], config["max_x"],
                                            config["min_y"],config["max_y"]).transpose(1, 2)
    torch.save(ret_traj, "./data/120_256_256_10hz_posi/sample_traj.pt")

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')


    args = parser.parse_args()


    
    args.steps = 0

    main(args)
