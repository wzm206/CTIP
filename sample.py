import numpy as np
import torch
import torch.nn.functional as F
import yaml
import time,datetime
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from model.CTIP import CTIPModel
from dataset.carla import get_Carla_loader
from utils import *
import tqdm

from tensorboardX import SummaryWriter

untransform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                         std=[1/0.229, 1/0.224, 1/0.225]),
])

def sample_waypoints(data_loader, config):
    loader_tqdm_test = tqdm.tqdm(data_loader, desc="Test Batch", leave=False)
    ret_waypoint = None
    for i_index, (obs_images, waypoint, img_position) in enumerate(loader_tqdm_test):
        now_waypoint = waypoint_normalize(waypoint.to(config["device"]), 
                                                config["min_x"], config["max_x"],
                                                config["min_y"],config["max_y"]).transpose(1, 2)
        if i_index==0:
            ret_waypoint = now_waypoint
            continue
        ret_waypoint = torch.cat([ret_waypoint, now_waypoint], dim=0)
        if i_index>=4:
            break
    return

def get_scores(data_loader, model, config):
    loader_tqdm_test = tqdm.tqdm(data_loader, desc="Test Batch", leave=False)
    model.eval()
    with torch.no_grad():
        loss = 0
        for obs_images, waypoint, img_position in loader_tqdm_test:
            batch_size = obs_images.shape[0]
            batch_data = {}
            waypoint = waypoint.to(config["device"])
            sigle_img = obs_images[0,0].to(config["device"])
            chanel, w, h = sigle_img.shape
            same_imgs = sigle_img.expand(batch_size, chanel, w, h )
            batch_data["image"] = same_imgs
            batch_data["traj"] = waypoint_normalize(waypoint, 
                                                    config["min_x"], config["max_x"],
                                                    config["min_y"],config["max_y"]).transpose(1, 2)
            batch_score = model.get_score(batch_data)
            
            show_waypoint(sigle_img, batch_score, waypoint, config)


def show_waypoint(sigle_img, batch_score, all_waypoints, config):
    _, top5_index = torch.topk(batch_score, k=5, dim=0, largest=True, sorted=True)  # k=2
    _, last5_index = torch.topk(batch_score, k=5, dim=0, largest=False, sorted=True)  # k=2
    top5_data = to_numpy(torch.index_select(all_waypoints, dim=0, index=top5_index))
    last5_data = to_numpy(torch.index_select(all_waypoints, dim=0, index=last5_index))
    image_now_numpy = to_numpy(untransform(sigle_img))
    # visualize
    f, (all_ax) = plt.subplots(1, 2)
    all_ax[0].imshow(image_now_numpy.transpose(1,2,0))
    for i_waypoint_index, waypoint_now in enumerate(top5_data):
        all_ax[1].scatter(-waypoint_now[:,1],waypoint_now[:,0], c=[0.1, 0.2, 0.8], alpha=0.8)
        all_ax[1].plot(-waypoint_now[:,1],waypoint_now[:,0], c="b", alpha=0.8)
    for i_waypoint_index, waypoint_now in enumerate(last5_data):
        all_ax[1].scatter(-waypoint_now[:,1],waypoint_now[:,0], c=[0.8, 0.2, 0.1], alpha=0.8)
        all_ax[1].plot(-waypoint_now[:,1],waypoint_now[:,0], c="r", alpha=0.8)
    
    all_ax[1].set_xlim([config["min_y"], config["max_y"]])
    all_ax[1].set_ylim([config["min_x"], config["max_x"]])
    plt.show()
    plt.close()



def main(args):
    
    with open("config/carla.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        


    train_loader, test_loader = get_Carla_loader(config)

    model = CTIPModel().to(config["device"])
    model = load_model_para(model, config)

    # sample_waypoints(test_loader, config)
    # 这里的 waypoints 没有归一化
    sigle_img, batch_score, all_waypoints = get_scores(test_loader, model, config)
    
    
    

    



   
if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')


    args = parser.parse_args()
    nowtime = time.strftime("%m_%d_%H_%M_%S")
    args.log_file_name = nowtime



    
    args.steps = 0

    main(args)
