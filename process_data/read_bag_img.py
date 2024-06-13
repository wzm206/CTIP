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
input_dir = "/home/wzm/data/real_casia/exam_casia_day"
output_dir = "/home/wzm/data/real_casia/exam_casia_day_all_img"
imtopic = "/camera/color/image_raw/compressed"
delta_t = 1.0

# iterate recurisively through all the folders and get the path of files with .bag extension in the args.input_dir
bag_files = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".bag"):
            bag_files.append(os.path.join(root, file))

# processing loop
for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
    try:
        bag = rosbag.Bag(bag_path)
    except rosbag.ROSBagException as e:
        print(e)
        print(f"Error loading {bag_path}. Skipping...")
        continue
    file_name=bag_path.split("/")[-1].split(".")[0]
    os.mkdir(os.path.join(output_dir, file_name))
    # get start time of bag in seconds
    currtime = bag.get_start_time()
    starttime = currtime

    for topic, msg, t in bag.read_messages(topics=[imtopic]):
        if t.to_sec()-currtime>delta_t:
            currtime = t.to_sec()
            img = Image.open(io.BytesIO(msg.data))
            img.save(os.path.join(output_dir, file_name, str(currtime)+".jpg"))
            