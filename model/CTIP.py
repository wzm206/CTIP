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

class CTIPModel(nn.Module):
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


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# model = CLIPModel()
# batch = {}
# fake_img = torch.rand([11, 3, 224, 224])
# fake_traj = torch.rand([11, 2, 16])
# batch["image"] = fake_img
# batch["traj"] = fake_traj
# output = model(batch)
# print(output)