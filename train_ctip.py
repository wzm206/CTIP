import numpy as np
import torch
import torch.nn.functional as F
import yaml
import time,datetime

from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from model.CTIP import CTIPModel
from dataset.ctip_dataset import get_CTIP_loader
from utils import *
import matplotlib.pyplot as plt
import tqdm

from tensorboardX import SummaryWriter

untransform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                         std=[1/0.229, 1/0.224, 1/0.225]),
])

def train(data_loader, model, optimizer, config, args, writer): 
    loader_tqdm_train = tqdm.tqdm(data_loader, desc="Train Batch", leave=False)
    model.train()
    for i, (obs_images, waypoint, img_position) in enumerate(loader_tqdm_train):
        batch_data = {}
        tagets = model.get_targets(waypoint).to(config["device"])
        # print(tagets.mean())
        batch_data["image"] = obs_images[:,0].to(config["device"])
        batch_data["traj"] = waypoint_normalize(waypoint, config).transpose(1, 2).to(config["device"])

        optimizer.zero_grad()
        loss = model(batch_data, tagets)
        loss.backward()
        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)
        optimizer.step()
        args.steps += 1
        loader_tqdm_train.set_postfix(now_batch_loss=loss.item())
        if i<3:
            generate_samples(batch_data, model, config, sample_name = "train")

def test(data_loader, model, config, args, writer):
    model.eval()
    loader_tqdm_test = tqdm.tqdm(data_loader, desc="Test Batch", leave=False)
    with torch.no_grad():
        loss = 0
        for i, (obs_images, waypoint, img_position) in enumerate(loader_tqdm_test):
            batch_data = {}
            tagets = model.get_targets(waypoint).to(config["device"])
            batch_data["image"] = obs_images[:,0].to(config["device"])
            batch_data["traj"] = waypoint_normalize(waypoint, config).transpose(1, 2).to(config["device"])
            now_loss = model(batch_data, tagets).item()
            loss += now_loss
            loader_tqdm_test.set_postfix(now_batch_loss=loss.item())
            if i<3:
                generate_samples(batch_data, model, config, sample_name = "test")
            
        loss /= len(data_loader)
        # Logs
        writer.add_scalar('loss/test', loss.item(), args.steps)
    return loss
def generate_samples(batch, model, config, sample_name = "aaa"):
    with torch.no_grad():
        all_waypoints = waypoint_unnormalize(batch["traj"].transpose(1, 2), config)
        sigle_img = batch["image"][0]
        batch_score = model.get_score(batch)
        _, top5_index = torch.topk(batch_score, k=5, dim=1, largest=True, sorted=True)  # k=2
        _, last5_index = torch.topk(batch_score, k=5, dim=1, largest=False, sorted=True)  # k=2
        top5_data = to_numpy(torch.index_select(all_waypoints, dim=0, index=top5_index[0]))
        last5_data = to_numpy(torch.index_select(all_waypoints, dim=0, index=last5_index[0]))
        image_now_numpy = to_numpy(untransform(sigle_img))
        gt_data = to_numpy(all_waypoints[0])
        # visualize
        f, (all_ax) = plt.subplots(1, 2)
        all_ax[0].imshow(image_now_numpy.transpose(1,2,0))
        # gt
        all_ax[1].plot(-gt_data[:,1],gt_data[:,0], c="g", alpha=0.8, linewidth = 5)
        for i_waypoint_index, waypoint_now in enumerate(top5_data):
            all_ax[1].plot(-waypoint_now[:,1],waypoint_now[:,0], c="b", alpha=0.3)
        for i_waypoint_index, waypoint_now in enumerate(last5_data):
            all_ax[1].plot(-waypoint_now[:,1],waypoint_now[:,0], c="r", alpha=0.3)
        
        all_ax[1].set_xlim([config["y_min"]-3, config["y_max"]+3])
        all_ax[1].set_ylim([config["x_min"], config["x_max"]+5])
        nowtime = time.strftime("%m_%d_%H_%M_%S")
        plt.savefig(os.path.join(config["log_samples"], sample_name+nowtime+".jpg"))
        plt.close()

def main(args):
    
    with open("config/ctip.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open('./logs/{0}/train_config.yaml'.format(args.log_file_name), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter('./logs/{0}'.format(args.log_file_name))
    save_filename = './logs/{0}/models'.format(args.log_file_name)
    config["log_samples"] = './logs/{0}/samples'.format(args.log_file_name)
    train_loader, test_loader = get_CTIP_loader(config)

    model = CTIPModel().to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))

    best_loss = -1.
    for epoch in range(config["epochs"]):
        train(train_loader, model, optimizer, config, args, writer)
        loss = test(test_loader, model, config, args, writer)
        print(epoch)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)
            
        # 查看网络梯度
        # for name, param in model.named_parameters():  # 返回网络的
        #     writer.add_histogram(name + '_grad', param.grad, epoch)
        #     writer.add_histogram(name + '_data', param, epoch)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')


    args = parser.parse_args()
    nowtime = time.strftime("%m_%d_%H_%M_%S")
    args.log_file_name = nowtime

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists(f'./logs/{nowtime}/models'):
        os.makedirs(f'./logs/{nowtime}/models')
    if not os.path.exists(f'./logs/{nowtime}/samples'):
        os.makedirs(f'./logs/{nowtime}/samples')    
    print(f'./logs/{nowtime}/models')


    
    args.steps = 0

    main(args)
