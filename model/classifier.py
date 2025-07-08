import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
