import random
import os

dataset_name = "bionic"
input_folder_dir = os.path.join("./data", dataset_name, "data")
    
print("all bags OK! Start split...")


# Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
folder_names = [
    f
    for f in os.listdir(input_folder_dir)
    if os.path.isdir(os.path.join(input_folder_dir, f))
    and "traj_data.pkl" in os.listdir(os.path.join(input_folder_dir, f))
]

# Randomly shuffle the names of the folders
random.shuffle(folder_names)

# Split the names of the folders into train and test sets
split_index = int(0.9 * len(folder_names))
train_folder_names = folder_names[:split_index]
test_folder_names = folder_names[split_index:]

# Write the names of the train and test folders to files
with open(os.path.join("./data", dataset_name,"train_traj_names.txt"), "w") as f:
    for folder_name in train_folder_names:
        f.write(folder_name + "\n")

with open(os.path.join("./data", dataset_name,"test_traj_names.txt"), "w") as f:
    for folder_name in test_folder_names:
        f.write(folder_name + "\n")