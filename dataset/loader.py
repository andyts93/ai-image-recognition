import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import PIL
import io

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

def get_dataloader(tar_pattern, batch_size, num_workers, shuffle=True):
    dataset = (
        wds.WebDataset(tar_pattern, resampled=shuffle, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(image_transform, parse_class)
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)