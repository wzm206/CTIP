import os
import pickle
from PIL import Image
import io
import argparse
import tqdm
import yaml
import rosbag
import sys
import random

sys.path.append(os.getcwd())
print("now work space",sys.path)
# utils

from process_data.utils import *


def main(args: argparse.Namespace):

    # load the config file
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # iterate recurisively through all the folders and get the path of files with .bag extension in the args.input_dir
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
    if args.num_bags >= 0:
        bag_files = bag_files[: args.num_bags]

    # processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            b = rosbag.Bag(bag_path)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # name is that folders separated by _ and then the last part of the path
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]

        # load the hdf5 file
        
        get_ctip_images_and_odom(
            bag=b,
            config=config,
            traj_name=traj_name,
            output_path=os.path.join(args.output_dir, "data")
            
        )
  
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # number of trajs to process
    parser.add_argument(
        "--num-bags",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=-1,
        type=float,
        help="how many data to store once, -1 is process all",
    )

    parser.add_argument(
        "--config_path",
        "-conpath",
        default="config/ctip.yaml",
        type=str,
        help="how many data to store once, -1 is process all",
    )
    parser.add_argument(
        "--split", "-sp", type=float, default=0.9, help="Train/test split (default: 0.8)"
    )
    args = parser.parse_args()
    # all caps for the dataset name
    print(f"STARTING PROCESSING DATASET")
    print(args)
    main(args)
    print(f"FINISHED PROCESSING DATASET")
