import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import pickle

transform = transforms.Compose([
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

def image_transform(img):
    return transform(img)

def parse_part_id(part_id):
    return part_id.decode('utf-8')

class TripletDataset(Dataset):
    def __init__(self, tar_pattern, part_id_map_path, min_images_per_part=3):
        with open(part_id_map_path, 'rb') as f:
            self.part_id_map = pickle.load(f)

        self.part_id_map = {
            k: v for k, v in self.part_id_map.items() if len(v) >= min_images_per_part
        }
        self.valid_part_ids = list(self.part_id_map.keys())

        self.dataset = (
            wds.WebDataset(tar_pattern, handler=wds.ignore_and_continue, empty_check=False, shardshuffle=False)
            .decode('pil')
            .to_tuple('jpg', 'pid')
            .map_tuple(image_transform, parse_part_id)
        )

        self.samples = []
        for jpg, pid in self.dataset:
            if pid in self.valid_part_ids:
                self.samples.append([jpg, pid])

    def __len__(self):
        return len(self.valid_part_ids)
    
    def __getitem__(self, idx):
        anchor_part = self.valid_part_ids[idx]
        anchor_keys = self.part_id_map[anchor_part]

        anchor_img, positive_img = random.sample(
            [img for img, pid in self.samples if pid == anchor_part], 2
        )

        negative_part = random.choice([p for p in self.valid_part_ids if p != anchor_part])
        negative_img = random.choice(
            [img for img, pid in self.samples if pid == negative_part]
        )

        return anchor_img, positive_img, negative_img

def get_triplet_dataloader(tar_pattern, part_id_map_path, batch_size, num_workers):
    dataset = TripletDataset(tar_pattern, part_id_map_path)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)