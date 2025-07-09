import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x