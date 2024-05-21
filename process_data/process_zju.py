import os
import numpy as np
import sys
import pickle
sys.path.append(os.getcwd())
from process_data.utils import rot2euler
from PIL import Image  


base_path = "/home/wzm/data/CTIP_data/zju/all_files"
file_name_list = os.listdir(base_path)

for now_file_name in file_name_list:

    ori_path = os.path.join(base_path, now_file_name)
    output_path = "/home/wzm/project/wzm/CTIP/data/zju/data"


    traj_name = now_file_name
    icp_path = os.path.join(ori_path, "icp", "icp.txt")
    img_path = os.path.join(ori_path, "image_00", "data")

    img_path_list = os.listdir(img_path)
    icp_data = np.loadtxt(icp_path)    

    cut = False
    synced_imdata = []
    xys = []
    yaws = []

    curr_imdata = None
    curr_odomdata = None
    traj_index = 0
    for i, now_icp in enumerate(icp_data):
        if i<20 or i%2!=0:
            continue
        x, y = now_icp[3], now_icp[7]
        x_last, y_last = icp_data[i-10][3], icp_data[i-10][7]
        curr_img_name = str(i).zfill(10)+".jpg" # 数字转化为字符串
        curr_img_path = os.path.join(ori_path, "image_00", "data", curr_img_name)
        now_rot = np.array([now_icp[0:3], now_icp[4:7], now_icp[8:11]])
        _, _, yaw = rot2euler(now_rot)

        if abs(x-x_last)+abs(y-y_last)<0.05:
            cut = True
        curr_imdata = Image.open(curr_img_path).resize((224, 224))
        synced_imdata.append(curr_imdata)
        xys.append([x,y])
        yaws.append(yaw)

        # 及时截断，有cut说明有碰撞信号了

        if len(synced_imdata)>999:
            cut = True
        
        if cut:
            if len(synced_imdata) > 50:
                img_data = synced_imdata
                traj_data={}
                traj_data["position"]=np.array(xys)
                traj_data["yaw"]=np.array(yaws)
                assert len(img_data)==len(xys)
                
                # 有很多段小轨迹，这是其中的一段
                img_data_i = img_data
                traj_data_i = traj_data
                traj_name_i = traj_name + f"_{traj_index}"
                traj_index += 1
                
                traj_folder_i = os.path.join(output_path, traj_name_i)
                # make a folder for the traj
                if not os.path.exists(traj_folder_i):
                    os.makedirs(traj_folder_i)
                with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                    traj_data_dic = {"traj_name":traj_name_i, "traj_data":traj_data_i}
                    pickle.dump(traj_data_dic, f)
                # save the image data to disk
                for i, img in enumerate(img_data_i):
                    img.save(os.path.join(traj_folder_i, f"{i}.jpg"))
                
                
                
            cut = False
            curr_imdata = None
            curr_odomdata = None
            synced_imdata.clear()
            xys.clear()
            yaws.clear()

    
