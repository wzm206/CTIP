import torch
import torch.nn as nn
import torchvision
import torch.functional as F

class TrajEncoder(nn.Module):
    def __init__(self, waypoint_dim=16, dropout=0.1):
        super().__init__()
        self.projection1 = nn.Linear(waypoint_dim, 64)
        self.projection2 = nn.Linear(64, 128)
        self.projection3 = nn.Linear(128, 256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(256)

    def forward(self, x):
        x = self.gelu(self.projection1(x))
        x = self.gelu(self.projection2(x))
        x = self.dropout(self.projection3(x))
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

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=1,
        image_embedding=1000,
        traj_embedding=256,
    ):
        super().__init__()
        self.image_encoder = torchvision.models.resnet18(pretrained=True)
        self.traj_encoder = TrajEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.traj_projection = ProjectionHead(embedding_dim=traj_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        traj_features = self.traj_encoder(batch["traj"])
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        traj_embeddings = self.traj_projection(traj_features)

        print(image_embeddings.size())
        print(traj_embeddings.size())
        
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

model = CLIPModel()
batch = {}
fake_img = torch.rand([11, 3, 224, 224])
fake_traj = torch.rand([11, 2, 16])

batch["image"] = fake_img
batch["traj"] = fake_traj
output = model(batch)
print(output.size())