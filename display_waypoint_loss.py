import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml

with open("config/carla.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
traj_path = "./data/120_256_256_10hz_posi/sample_traj.pt"
traj_dic = torch.load(traj_path)

waypoint_ori_train = traj_dic["waypoint_ori_train"] 
waypoint_normal_train = traj_dic["waypoint_normal_train"] 

print(waypoint_ori_train.size())

for i in range(20):
    index_rand = torch.randint(low=0,high=100,size=[2])
    now_waypoints = torch.index_select(waypoint_ori_train, dim=0, index=index_rand)

    # visualize
    traj1, traj2 = now_waypoints
    fig, ax = plt.subplots(facecolor ='#A0F0CC')
    loss = max(F.mse_loss(traj1, traj2, reduction='none').mean(dim=1))
    print(loss)
    ax.scatter(-traj1[:,1],traj1[:,0], c=[0.1, 0.2, 0.8], alpha=0.8)
    ax.plot(-traj1[:,1],traj1[:,0], c="b", alpha=0.8)
    ax.scatter(-traj2[:,1],traj2[:,0], c=[0.8, 0.2, 0.1], alpha=0.8)
    ax.plot(-traj2[:,1],traj2[:,0], c="r", alpha=0.8)
    
    ax.set_xlim([config["min_y"], config["max_y"]])
    ax.set_ylim([config["min_x"], config["max_x"]])
    plt.show()
    plt.close()