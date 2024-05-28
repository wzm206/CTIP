import random
import os
import yaml
import numpy as np
import tqdm
import pickle
import sys
sys.path.append(os.getcwd())

dataset_name = "SACSoN"
input_folder_dir = os.path.join("./data", dataset_name, "data")
    
print("all bags OK! Start split...")

with open("config/ctip.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
from dataset.data_utils import (
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

down_rate = config[dataset_name]["down_rate"]
data_folder_posi = os.path.join(config[dataset_name]["output_dir"], "data")
traj_names_file_train = os.path.join(config[dataset_name]["output_dir"], "train_traj_names.txt")
traj_names_file_test = os.path.join(config[dataset_name]["output_dir"], "test_traj_names.txt")

# 准备读取所有轨迹片段
# Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
folder_names = [
    f
    for f in os.listdir(input_folder_dir)
    if os.path.isdir(os.path.join(input_folder_dir, f))
    and "traj_data.pkl" in os.listdir(os.path.join(input_folder_dir, f))
]

image_size = config["image_size"]
len_traj_pred = config["len_traj_pred"]
context_size = config["context_size"]
waypoint_spacing = config[dataset_name]["waypoint_spacing"]
end_slack = config[dataset_name]["end_slack"]

index_data_dic = {}
index_data_dic["posi"] = []

print("process posi data")
for traj_name in tqdm.tqdm(folder_names, dynamic_ncols=True):
    # into one traj file

    with open(os.path.join(input_folder_dir, traj_name, "traj_data.pkl"), "rb") as f:
        traj_data = pickle.load(f)
    # format:
    # traj_data_dic = {"traj_name":traj_name_i, "traj_time":time_data_i, "traj_data":traj_data_i}
    # traj_data_i = {"position": np.array(xys), "yaw": np.array(yaws)}
    traj_len = len(traj_data["traj_data"]["position"])
    begin_index = context_size * waypoint_spacing
    end_index = traj_len - len_traj_pred * waypoint_spacing - end_slack

    for curr_index in range(begin_index, end_index, down_rate):
        sigle_data_dic = {}
        img_index = range(curr_index, curr_index-context_size*waypoint_spacing, -waypoint_spacing)
        sigle_data_dic["img_index"] = img_index
        traj_index = range(curr_index, curr_index+len_traj_pred*waypoint_spacing, waypoint_spacing)
        sigle_data_dic["traj_index"] = traj_index
        now_position, now_yaw = traj_data["traj_data"]["position"][curr_index], traj_data["traj_data"]["yaw"][curr_index]
        abs_position_pre = traj_data["traj_data"]["position"][traj_index]
        sigle_data_dic["position_predict_data"] = to_local_coords(abs_position_pre, now_position, now_yaw)
        # sigle_data_dic["time_predict_data"] = traj_data["traj_time"][traj_index]
        abs_position_img =traj_data["traj_data"]["position"][img_index]
        sigle_data_dic["position_img_data"] = to_local_coords(abs_position_img, now_position, now_yaw)
        # sigle_data_dic["time_img_data"] = traj_data["traj_time"][img_index]
        sigle_data_dic["traj_name"] = traj_data["traj_name"]
        index_data_dic["posi"].append(sigle_data_dic)
        
        
print("data has {} items".format(len(index_data_dic["posi"])) )

# all data in index_data_dic
# Randomly shuffle the names of the folders
random.shuffle(index_data_dic["posi"])

# Split the names of the folders into train and test sets
split_index = int(0.9 * len(index_data_dic["posi"]))
train_folder_names = index_data_dic["posi"][:split_index]
test_folder_names = index_data_dic["posi"][split_index:]

# Write the names of the train and test folders to files

with open(os.path.join("./data", dataset_name,"train_traj_names.pkl"), "wb") as f:
    pickle.dump(train_folder_names, f)
with open(os.path.join("./data", dataset_name,"test_traj_names.pkl"), "wb") as f:
    pickle.dump(test_folder_names, f)