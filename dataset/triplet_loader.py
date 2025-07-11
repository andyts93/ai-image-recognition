import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import pickle
from collections import defaultdict

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
    def __init__(self, tar_pattern, part_id_map_path, min_images_per_part=2):
        with open(part_id_map_path, 'rb') as f:
            self.part_id_map = pickle.load(f)

        self.part_id_map = {
            k: v for k, v in self.part_id_map.items() if len(v) >= min_images_per_part
        }
        self.valid_part_ids = list(self.part_id_map.keys())

        # Costruisci cls → [part_id, part_id, ...]
        self.cls_to_part_ids = defaultdict(list)
        self.part_id_to_cls = {}

        for part_id, entries in self.part_id_map.items():
            cls = entries[0][1]  # prendi la classe dal primo elemento
            self.cls_to_part_ids[cls].append(part_id)
            self.part_id_to_cls[part_id] = cls

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
        anchor_cls = self.part_id_to_cls[anchor_part]

        anchor_img, positive_img = random.sample(
            [img for img, pid in self.samples if pid == anchor_part], 2
        )

        # Negativo: diverso part_id ma stessa classe
        candidate_neg_parts = [
            pid for pid in self.cls_to_part_ids[anchor_cls] if pid != anchor_part
        ]

        if candidate_neg_parts:
            negative_part = random.choice(candidate_neg_parts)
        else:
            # fallback: diverso part_id, qualsiasi classe
            negative_part = random.choice([pid for pid in self.valid_part_ids if pid != anchor_part])

        negative_imgs = [img for img, pid in self.samples if pid == negative_part]
        negative_img = random.choice(negative_imgs)

        return anchor_img, positive_img, negative_img

def get_triplet_dataloader(tar_pattern, part_id_map_path, batch_size, num_workers):
    dataset = TripletDataset(tar_pattern, part_id_map_path)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)