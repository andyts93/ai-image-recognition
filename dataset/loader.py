import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import PIL
import io
from torchvision.transforms import ToTensor
to_tensor = ToTensor()
import torch

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def parse_class(x):
    return int(x)

def image_transform(img):
    return transform(img)

def collate_fn_debug(batch):
    imgs, labels = zip(*batch)
    imgs = [to_tensor(img) for img in imgs]
    labels = list(labels)
    return torch.stack(imgs), torch.tensor(labels)

def get_dataloader(tar_pattern, batch_size, num_workers, shuffle=True):
    if (shuffle):
        shardshuffle = 100
    else:
        shardshuffle = False

    webds = wds.WebDataset(tar_pattern, resampled=False, shardshuffle=shardshuffle, empty_check=False)

    if (shuffle):
        webds.shuffle(1000)

    dataset = (
        webds
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(image_transform, parse_class)
    )

    if (shuffle):
        dataset.shuffle(1000)

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

def get_debug_dataloader(tar_pattern, batch_size, num_workers):
    dataset = (
        wds.WebDataset(tar_pattern, resampled=False, empty_check=False)
        .decode("pil")
        .to_tuple("jpg", "cls")
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn_debug)  