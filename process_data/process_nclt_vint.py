
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image as Image
import pickle
# 20880, 612861
def main():

    base_path = "/home/wzm/project/wzm/CTIP/data/nclt/data"
    date_list = os.listdir(base_path)
    output_path = "/home/wzm/project/wzm/vint_wzm/data/nclt"
    

    
    for file_name in date_list:
        with open(os.path.join(base_path, file_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        
        with open(os.path.join(output_path, file_name, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data["traj_data"], f)
        
        

if __name__ == '__main__':
    main()
