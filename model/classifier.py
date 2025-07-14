# import torch.nn as nn
# from torchvision.models import resnet50, ResNet50_Weights

# def get_model(num_classes):
#     model = resnet50(weights=ResNet50_Weights.DEFAULT)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def get_model(num_classes):
    # Carica il modello pre-addestrato
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    
    # Sostituisci il classificatore finale. 
    # NOTA: in EfficientNet il layer si chiama 'classifier', non 'fc'.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model