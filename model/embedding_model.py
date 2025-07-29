import torch
import torch.nn as nn
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

import timm  # Assicurati di averlo installato: pip install timm


class EmbeddingNet(nn.Module):
    """
    Nuova versione del modello di embedding che usa un Vision Transformer (ViT) come backbone.
    """

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        # Carica un Vision Transformer (ViT) pre-addestrato su ImageNet
        # 'num_classes=0' rimuove la testa di classificazione originale.
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0
        )

        # Ottiene la dimensione delle feature dal backbone del ViT
        backbone_output_features = self.backbone.embed_dim

        # Aggiungi una nuova "testa" per proiettare le feature nella dimensione di embedding desiderata
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(backbone_output_features),  # Normalizzazione specifica per i Transformer
            nn.Linear(backbone_output_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Passa l'immagine attraverso il backbone ViT
        x = self.backbone(x)
        # Passa le feature attraverso la testa di embedding
        x = self.embedding_head(x)
        return x