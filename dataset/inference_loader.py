import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

def parse_pid(x):
    return x.decode("utf-8")

def get_inference_dataloader(tar_pattern, batch_size, num_workers, shuffle=True):
    dataset = wds.WebDataset(tar_pattern, empty_check=False, shardshuffle=False)

    if shuffle:
        dataset = dataset.shuffle(10000)

    dataset = (
        dataset
        .decode("pil")
        .to_tuple("jpg", "cls", "pid")
        .map_tuple(image_transform, parse_class, parse_pid)
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)