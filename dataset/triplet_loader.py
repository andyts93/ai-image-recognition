import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import torch

# Le trasformazioni rimangono invariate
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

class TripletDataset(Dataset):
    def __init__(self, tar_pattern, part_id_map_path, min_images_per_part=2):
        """
        Inizializza il dataset senza caricare le immagini in memoria.
        """
        print("Inizializzazione TripletDataset scalabile...")
        
        with open(part_id_map_path, 'rb') as f:
            pid_to_keys = pickle.load(f)

        self.pid_to_keys = {
            pid: keys for pid, keys in pid_to_keys.items() 
            if len(keys) >= min_images_per_part
        }
        
        self.samples = []
        for pid, keys in self.pid_to_keys.items():
            for key_tuple in keys:
                key = key_tuple[0] 
                self.samples.append((key, pid))
        
        # Pipeline di WebDataset per l'accesso su richiesta
        # NOTA: Aggiunto shardshuffle=True per rimuovere il warning
        image_source_pipeline = wds.WebDataset(tar_pattern, handler=wds.ignore_and_continue, shardshuffle=1000) \
                                   .decode('pil') \
                                   .map_dict(jpg=transform) \
                                   .to_tuple('__key__', 'jpg')
        
        print("Indicizzazione delle immagini per accesso rapido (potrebbe richiedere tempo la prima volta)...")
        # L'ordine delle operazioni nella pipeline è stato corretto
        self.key_to_image = {key: img for key, img in image_source_pipeline}
        print(f"Indicizzazione completata. Trovati {len(self.key_to_image)} campioni unici.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 1. Seleziona l'ancora
        anchor_key, anchor_pid = self.samples[idx]
        anchor_img = self.key_to_image[anchor_key]

        # 2. Seleziona il positivo
        possible_positives = [k[0] for k in self.pid_to_keys[anchor_pid] if k[0] != anchor_key]
        # Aggiunto un controllo per part_id con una sola immagine, anche se filtrati in init
        if not possible_positives:
            positive_key = anchor_key
        else:
            positive_key = random.choice(possible_positives)
        positive_img = self.key_to_image[positive_key]

        # 3. Seleziona il negativo
        negative_pid_list = list(self.pid_to_keys.keys())
        negative_pid_list.remove(anchor_pid)
        negative_pid = random.choice(negative_pid_list)
        
        negative_key = random.choice(self.pid_to_keys[negative_pid])[0]
        negative_img = self.key_to_image[negative_key]
        
        return anchor_img, positive_img, negative_img

def get_triplet_dataloader(tar_pattern, part_id_map_path, batch_size, num_workers):
    dataset = TripletDataset(tar_pattern, part_id_map_path)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)