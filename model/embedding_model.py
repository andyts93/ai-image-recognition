import torch
# import torch.nn as nn
# from torchvision.models import resnet50, ResNet50_Weights
#
# class EmbeddingNet(nn.Module):
#     def __init__(self, embedding_dim=128):
#         super().__init__()
#         base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
#         self.backbone = nn.Sequential(*list(base_model.children())[:-1])
#         self.embedding = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(base_model.fc.in_features, embedding_dim),
#             nn.BatchNorm1d(embedding_dim),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.embedding(x)
#         return x

import torch
import torch.nn as nn
import timm
from .arcface import ArcFace  # Importa il layer che abbiamo appena creato


class EmbeddingNetArcFace(nn.Module):
    def __init__(self, embedding_dim, num_classes, pretrained=True):
        super().__init__()
        # Carica il backbone ViT
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        backbone_output_features = self.backbone.embed_dim

        # Testa di embedding che produce il vettore finale
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(backbone_output_features),
            nn.Linear(backbone_output_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)  # Rimuoviamo ReLU per ArcFace
        )

        # Layer ArcFace, che agisce sull'embedding
        self.arcface_head = ArcFace(in_features=embedding_dim, out_features=num_classes, s=30.0, m=0.5)

    def forward(self, x, label=None):
        # Estrai le feature dal backbone
        features = self.backbone(x)
        # Calcola l'embedding finale
        embedding = self.embedding_head(features)

        # Se siamo in fase di training, calcola l'output di ArcFace per la loss
        if label is not None:
            return self.arcface_head(embedding, label)

        # Se siamo in fase di inferenza, restituisci solo l'embedding
        return embedding