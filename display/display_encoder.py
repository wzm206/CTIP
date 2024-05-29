import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time,datetime
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import sys
import os
sys.path.append(os.getcwd())
from model.CTIP import CTIPModel
from dataset.ctip_dataset import get_CTIP_loader_from_list
from utils import *
import tqdm

from tensorboardX import SummaryWriter

untransform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                         std=[1/0.229, 1/0.224, 1/0.225]),
])

# def test(data_loader, model, config):
#     model.eval()
#     loader_tqdm_test = tqdm.tqdm(data_loader, desc="Test Batch", leave=False)
#     with torch.no_grad():
#         loss = 0
#         for i, (obs_images, waypoint, img_position) in enumerate(loader_tqdm_test):
#             batch_data = {}
#             tagets = model.get_targets(waypoint, config).to(config["device"])
#             batch_data["image"] = obs_images[:,0].to(config["device"])
#             batch_data["traj"] = waypoint_normalize(waypoint, config).transpose(1, 2).to(config["device"])
#             loss += model(batch_data, tagets)
            
#             generate_samples(batch_data, model, config, sample_name = "sp")
            
#         loss /= len(data_loader)

#     return loss
# def generate_samples(batch, model, config, sample_name = "aaa"):
#     with torch.no_grad():
#         all_waypoints = waypoint_unnormalize(batch["traj"].transpose(1, 2), config)
#         sigle_img = batch["image"][0]
#         batch_score = model.get_score(batch)
#         _, top5_index = torch.topk(batch_score, k=5, dim=1, largest=True, sorted=True)  # k=2
#         _, last5_index = torch.topk(batch_score, k=5, dim=1, largest=False, sorted=True)  # k=2
#         top5_data = to_numpy(torch.index_select(all_waypoints, dim=0, index=top5_index[0]))
#         last5_data = to_numpy(torch.index_select(all_waypoints, dim=0, index=last5_index[0]))
#         image_now_numpy = to_numpy(untransform(sigle_img))
#         gt_data = to_numpy(all_waypoints[0])
#         # visualize
#         f, (all_ax) = plt.subplots(1, 2)
#         all_ax[0].imshow(image_now_numpy.transpose(1,2,0))
#         # gt
#         all_ax[1].plot(-gt_data[:,1],gt_data[:,0], c="g", alpha=0.8, linewidth = 5)
#         for i_waypoint_index, waypoint_now in enumerate(top5_data):
#             all_ax[1].plot(-waypoint_now[:,1],waypoint_now[:,0], c="b", alpha=0.3)
#         for i_waypoint_index, waypoint_now in enumerate(last5_data):
#             all_ax[1].plot(-waypoint_now[:,1],waypoint_now[:,0], c="r", alpha=0.3)
        
#         all_ax[1].set_xlim([config["y_min"], config["y_max"]])
#         all_ax[1].set_ylim([config["x_min"], config["x_max"]+5])
#         nowtime = time.strftime("%m_%d_%H_%M_%S")
#         plt.savefig(os.path.join("./samples", sample_name+nowtime+".jpg"))
#         plt.close()

def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

def main(args):
    
    with open("config/ctip.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    train_loader, test_loader = get_CTIP_loader_from_list(config, dataset_name_list=["bionic"])
    model = CTIPModel().to(config["device"])
    model = load_model_para(model, config)
    # test(train_loader, model, config)
    # print(model)
    resnet_encoder = list(list(model.children())[1].children())
    # print(resnet_encoder)
    new_model = nn.Sequential(*resnet_encoder[:8])
    new_model.eval()
    loader_tqdm_train = tqdm.tqdm(train_loader, desc="Test Batch", leave=False)
    for i, (obs_images, waypoint, img_position) in enumerate(loader_tqdm_train):
        with torch.no_grad():
            obs_images = obs_images[:,0].to(config["device"])
            output = new_model(obs_images)
            output_mean = torch.mean(output, dim=1)
            heat_map_ori = cv2.resize(output_mean[0].cpu().numpy(), (224, 224))
            heat_map_ori = cv2.convertScaleAbs(heat_map_ori,alpha=100,beta=0)
            heatmap_color = cv2.applyColorMap(heat_map_ori, cv2.COLORMAP_JET)
            ori_img = cv2.convertScaleAbs(untransform(obs_images)[0].cpu().numpy().transpose(1,2,0),alpha=255)
            alpha = 0.5 # 设置覆盖图片的透明度
            #cv2.rectangle (overlay, (0, 0), (merge_img.shape [1], merge_img.shape [0]), (0, 0, 0), -1) # 设置蓝色为热度图基本色
            ret = cv2.addWeighted (ori_img, alpha, heatmap_color, 1-alpha, 0) # 将背景热度图覆盖到原图
            cv2.imwrite(f"./display/samples/{i}.jpg", ret)
            
            # cv2.imshow("ret", ret)
            # cv2.waitKey(0)




   
if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')


    args = parser.parse_args()
    nowtime = time.strftime("%m_%d_%H_%M_%S")
    args.log_file_name = nowtime
    main(args)
