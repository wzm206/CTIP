
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image as Image
import pickle
# 20880, 612861
def main():
    file_name_list = ["2013-04-05", "2012-05-26", "2012-06-15", "2012-08-04", "2012-08-20"]
    base_path = "/home/wzm/data/CTIP_data/nclt"
    output_path = "/home/wzm/project/wzm/CTIP/data/nclt/data"
    for file_name in file_name_list:
        image_path = os.path.join(base_path, file_name, "Cam5")
        gt_path = os.path.join(base_path, "groundtruth_"+file_name+".csv")
        image_names = os.listdir(image_path)
        image_names.sort()
        
        # print((image_names[:20]))
        # t, x, y, z, r, p, y  1230,7
        gt = np.loadtxt(gt_path, delimiter = ",")
        print(gt.shape)
        
        cut = False
        synced_imdata = []
        xys = []
        yaws = []

        curr_imdata = None
        curr_odomdata = None
        traj_index = 0
        traj_name = file_name
        
        # (1232, 1616) -> 1232, 924
        for loop_index, image_name_tiff in enumerate(image_names):
            if loop_index<50 or loop_index%2!=0:
                continue
            
            time_now = int(image_name_tiff[:-5])
            ori_image = Image.open(os.path.join(image_path, image_name_tiff)).rotate(270, expand = 1)
            curr_imdata = Image.Image.crop(ori_image, (0, 346, 1232, 1616-346)).resize((224,224))
            x = np.interp(time_now, gt[:, 0], gt[:, 1])
            y = np.interp(time_now, gt[:, 0], gt[:, 2])
            yaw = np.interp(time_now, gt[:, 0], gt[:, 6])
            


            x_last = np.interp(time_now-200000*5, gt[:, 0], gt[:, 1])
            y_last = np.interp(time_now-200000*5, gt[:, 0], gt[:, 2])

            if abs(x-x_last)+abs(y-y_last)<0.1:
                cut = True

            synced_imdata.append(curr_imdata)
            xys.append([x,-y])
            yaws.append(-yaw)  #这里坐标系不同

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
            
            if loop_index%500==0:
                print(loop_index)

if __name__ == '__main__':
    main()
