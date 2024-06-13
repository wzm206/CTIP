import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class TrajEncoder(nn.Module):
    def __init__(self, waypoint_dim=16, dropout=0.1):
        super().__init__()
        self.projection1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3)
        self.projection2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.projection3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=3)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.gelu(self.projection1(x))
        x = self.gelu(self.projection2(x))
        x = self.pool2(x)
        x = self.gelu(self.projection3(x))
        x = self.pool3(x).squeeze()
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CTIPModel_old(nn.Module):
    def __init__(
        self,
        temperature=1,
        image_embedding=1000,
        traj_embedding=256,
    ):
        super().__init__()
        self.image_encoder = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.traj_encoder = TrajEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.traj_projection = ProjectionHead(embedding_dim=128)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        traj_features = self.traj_encoder(batch["traj"])
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        traj_embeddings = self.traj_projection(traj_features)
        
        # Calculating the Loss
        logits = (traj_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = traj_embeddings @ traj_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
    def get_score(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        traj_features = self.traj_encoder(batch["traj"])
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        traj_embeddings = self.traj_projection(traj_features)
        
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        traj_embeddings = traj_embeddings / traj_embeddings.norm(dim=1, keepdim=True)
        
        # Calculating the Loss
        logits = (traj_embeddings @ image_embeddings.T) / self.temperature
        return logits[:, 0]


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()



class CTIPModel(nn.Module):
    def __init__(
        self,
        temperature=1,
        image_embedding=1000,
        traj_embedding=256,
        traj_error = 0.3,
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.image_encoder = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.traj_encoder = TrajEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.traj_projection = ProjectionHead(embedding_dim=128)
        self.temperature = temperature
        self.traj_error = traj_error
        
    def get_targets(self, waypoint_ori, config):
        threshold = float(config["threshold"])
        ori_traj = waypoint_ori.clone().detach()
        batch_size, length, chanel = ori_traj.shape
        clip_length = length//2
        ori_traj = ori_traj[:, clip_length:, 1]
        label_matrix = torch.zeros(batch_size, batch_size)
        for i in range(batch_size):
            now_traj = ori_traj[i]
            # 有1的行应该排除 
            dddd = torch.gt(ori_traj, now_traj-threshold) & torch.lt(ori_traj, now_traj+threshold)
            index = torch.any(dddd, dim=1)
            label_matrix[i]=index
        diag = torch.diag(label_matrix)
        
        return label_matrix - torch.diag_embed(diag)
            

    def forward(self, batch, targets):
        # target similar is 1
        # ([[0., 1., 1., 1., 0.],
        # [1., 0., 1., 1., 0.],
        # [1., 1., 0., 1., 1.],
        # [1., 1., 1., 0., 0.],
        # [0., 0., 1., 0., 0.]])
        
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        traj_features = self.traj_encoder(batch["traj"])
        batch_size, device = image_features.shape[0], image_features.device
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        traj_embeddings = self.traj_projection(traj_features)
        tagets_inverse = torch.where(targets==0, 1, 0)
        # Calculating the Loss
        logits = (traj_embeddings @ image_embeddings.T) / self.temperature
        logits = logits*tagets_inverse - 1000*targets
        labels = torch.arange(batch_size, device=device).long()
        
        # image_loss_mat = torch.cosine_similarity(image_embeddings.unsqueeze(1), image_embeddings.unsqueeze(0), dim=2)
        # image_loss_mat.diagonal().zero_()
        # image_loss_mat = image_loss_mat*tagets_inverse

        # traj_loss_mat = torch.cosine_similarity(traj_embeddings.unsqueeze(1), traj_embeddings.unsqueeze(0), dim=2)
        # traj_loss_mat.diagonal().zero_()
        # traj_loss_mat = traj_loss_mat*tagets_inverse
        
        loss = (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)
        ) / 2

        # return loss.mean()+traj_loss_mat.mean()+image_loss_mat.mean()
        return loss.mean()
    
    def get_score(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        traj_features = self.traj_encoder(batch["traj"])
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        traj_embeddings = self.traj_projection(traj_features)
        # Calculating the Loss
        logits = (traj_embeddings @ image_embeddings.T) / self.temperature
        return logits
    
    def get_score_deploy(self, sigle_img, traj_data):
        if sigle_img.shape[0] != 1:
            sigle_img = sigle_img.unsqueeze(0)
        # Getting Image and Text Features
        image_features = self.image_encoder(sigle_img)
        traj_features = self.traj_encoder(traj_data)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        traj_embeddings = self.traj_projection(traj_features)
        # Calculating the Loss
        logits = (traj_embeddings @ image_embeddings.T) / self.temperature
        return logits.squeeze()



# import matplotlib.pyplot as plt
# import yaml
# model = CTIPModel()
# batch = {}
# with open("config/carla.yaml", "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# traj_path = "./data/120_256_256_10hz_posi/sample_traj.pt"
# traj_dic = torch.load(traj_path)
# waypoint_ori_train = traj_dic["waypoint_ori_train"] 
# waypoint_normal_train = traj_dic["waypoint_normal_train"] 

# fake_img = torch.rand([5, 3, 224, 224])
# fake_traj = torch.rand([5, 2, 16])
# batch["image"] = fake_img
# batch["traj"] = waypoint_ori_train[5:10].cpu()
# torch.set_printoptions(threshold=1e5)
# tagets = model.get_targets(batch["traj"], config)

# input = {}
# print(batch["traj"].shape)
# input["traj"] = batch["traj"].transpose(1,2)
# input["image"] = fake_img
# model(input, tagets)
# # print(tagets)
# # tagets_inverse = torch.where(tagets==0, 1, 0)
# # print(tagets_inverse)
# # out = model(batch, tagets)
# fig, ax = plt.subplots(facecolor ='#A0F0CC')
# red_traj = batch["traj"][0]
# ax.scatter(-red_traj[:,1],red_traj[:,0], c=[0.8, 0.2, 0.1], alpha=0.8)
# ax.plot(-red_traj[:,1],red_traj[:,0], c="r", alpha=0.8, linewidth = 3)
# for i, traj in enumerate(batch["traj"]):
#     if tagets[0][i]==1:
#         continue
#     ax.scatter(-traj[:,1],traj[:,0], c=[0.1, 0.2, 0.8], alpha=0.8)
#     ax.plot(-traj[:,1],traj[:,0], c="b", alpha=0.8)
# ax.set_xlim([config["min_y"], config["max_y"]])
# ax.set_ylim([config["min_x"], config["max_x"]])
# plt.show()
# plt.close()







