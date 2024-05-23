import torch
import torch.nn.functional as F

a = torch.rand([4, 16])
b = torch.rand([5, 16])

simi = F.cosine_similarity(a.unsqueeze(1), a.unsqueeze(0), dim=2)

print(simi)
