import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

# Trasformazioni per l'addestramento (con augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Trasformazioni per la validazione (deterministica)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Usa Resize invece di RandomResizedCrop
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Carica la mappa una sola volta all'avvio del modulo
with open("data/dataset/part_id_to_idx.json", 'r') as f:
    part_id_to_idx = json.load(f)

def parse_pid(pid_bytes):
    """
    Converte il part_id originale (stringa) nel suo indice numerico (int).
    """
    pid_str = pid_bytes.decode('utf-8').strip()
    return part_id_to_idx.get(pid_str, -1) # Restituisce -1 se non trova il part_id (per sicurezza)

# def parse_pid(pid_bytes):
#     # Gestisce sia i pid numerici che stringa, per sicurezza
#     try:
#         return int(pid_bytes.decode('utf-8'))
#     except ValueError:
#         return pid_bytes.decode('utf-8')

def get_part_dataloader(tar_pattern, batch_size, num_workers):
    """ Dataloader per il training con data augmentation. """
    dataset = (
        wds.WebDataset(tar_pattern, resampled=False, shardshuffle=1000)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "pid")
        .map_tuple(train_transform, parse_pid)
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

def get_val_dataloader(tar_pattern, batch_size, num_workers):
    """ Dataloader per la validazione, senza augmentation né shuffling. """
    dataset = (
        wds.WebDataset(tar_pattern, resampled=False, shardshuffle=False) # No resample, no shuffle
        .decode("pil")
        .to_tuple("jpg", "pid")
        .map_tuple(val_transform, parse_pid)
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)