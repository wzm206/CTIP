import numpy as np
import torch
import torch.nn.functional as F
import yaml
import time,datetime

from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from model.CTIP import CTIPModel
from dataset.ctip_dataset import get_CTIP_loader
from utils import waypoint_normalize, waypoint_unnormalize
import tqdm

from tensorboardX import SummaryWriter

untransform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                         std=[1/0.229, 1/0.224, 1/0.225]),
])

def train(data_loader, model, optimizer, config, args, writer): 
    loader_tqdm_train = tqdm.tqdm(data_loader, desc="Train Batch", leave=False)
    model.train()
    for obs_images, waypoint, img_position in loader_tqdm_train:
        batch_data = {}
        tagets = model.get_targets(waypoint).to(config["device"])
        # print(tagets.mean())
        batch_data["image"] = obs_images[:,0].to(config["device"])
        # batch_data["traj"] = waypoint_normalize(waypoint.to(config["device"]), 
        #                                         config["min_x"], config["max_x"],
        #                                         config["min_y"],config["max_y"]).transpose(1, 2)
        batch_data["traj"] = waypoint.to(config["device"]).transpose(1, 2)
        optimizer.zero_grad()
        loss = model(batch_data, tagets)
        loss.backward()
        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)
        optimizer.step()
        args.steps += 1
        loader_tqdm_train.set_postfix(now_batch_loss=loss.item())

def test(data_loader, model, config, args, writer):
    model.eval()
    loader_tqdm_test = tqdm.tqdm(data_loader, desc="Test Batch", leave=False)
    with torch.no_grad():
        loss = 0
        for obs_images, waypoint, img_position in loader_tqdm_test:
            batch_data = {}
            tagets = model.get_targets(waypoint).to(config["device"])
            batch_data["image"] = obs_images[:,0].to(config["device"])
            # batch_data["traj"] = waypoint_normalize(waypoint.to(config["device"]), 
            #                                         config["min_x"], config["max_x"],
            #                                         config["min_y"],config["max_y"]).transpose(1, 2)
            batch_data["traj"] = waypoint.to(config["device"]).transpose(1, 2)
            loss += model(batch_data, tagets)
        loss /= len(data_loader)
        # Logs
        writer.add_scalar('loss/test', loss.item(), args.steps)
    return loss
def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde

def main(args):
    
    with open("config/ctip.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    writer = SummaryWriter('./logs/{0}'.format(args.log_file_name))
    save_filename = './logs/{0}/models'.format(args.log_file_name)

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
    print(f'./logs/{nowtime}/models')


    
    args.steps = 0

    main(args)
