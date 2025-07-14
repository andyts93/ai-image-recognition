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
        1. Carica la mappa part_id -> [__key__]
        2. Crea un elenco di tutti i campioni (ancore possibili) come (key, pid).
        3. Prepara un'origine dati WebDataset per poter caricare le immagini su richiesta.
        """
        print("Inizializzazione TripletDataset scalabile...")
        
        # 1. Carica la mappa PID -> Keys creata da create_triplet.py
        with open(part_id_map_path, 'rb') as f:
            pid_to_keys = pickle.load(f)

        # Filtra i part_id che hanno almeno 'min_images_per_part' immagini
        self.pid_to_keys = {
            pid: keys for pid, keys in pid_to_keys.items() 
            if len(keys) >= min_images_per_part
        }
        
        # 2. Crea un elenco di tutti i campioni disponibili come coppie (key, pid)
        self.samples = []
        for pid, keys in self.pid_to_keys.items():
            for key_tuple in keys:
                # key_tuple è ('__key__', 'cls_idx'), a noi serve solo la chiave
                key = key_tuple[0] 
                self.samples.append((key, pid))
        
        # 3. Prepara il WebDataset per l'accesso su richiesta tramite chiave
        # Questo NON carica nulla in memoria, prepara solo la pipeline
        self.image_source = wds.WebDataset(tar_pattern, handler=wds.ignore_and_continue) \
                               .decode('pil') \
                               .to_tuple('__key__', 'jpg') \
                               .map_dict(jpg=transform)
        
        # Crea un dizionario in memoria per un accesso rapido: key -> image_tensor
        print("Indicizzazione delle immagini per accesso rapido (potrebbe richiedere tempo la prima volta)...")
        # Questa è la parte più dispendiosa, ma necessaria per l'accesso casuale
        self.key_to_image = {key: img for key, img in self.image_source}
        print(f"Indicizzazione completata. Trovati {len(self.key_to_image)} campioni unici.")

    def __len__(self):
        # La lunghezza del dataset è il numero totale di possibili ancore
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 1. Seleziona l'ancora
        anchor_key, anchor_pid = self.samples[idx]
        anchor_img = self.key_to_image[anchor_key]

        # 2. Seleziona il positivo
        possible_positives = [k[0] for k in self.pid_to_keys[anchor_pid] if k[0] != anchor_key]
        positive_key = random.choice(possible_positives)
        positive_img = self.key_to_image[positive_key]

        # 3. Seleziona il negativo
        # Scegli un part_id diverso da quello dell'ancora
        negative_pid_list = list(self.pid_to_keys.keys())
        negative_pid_list.remove(anchor_pid)
        negative_pid = random.choice(negative_pid_list)
        
        # Scegli una chiave casuale da quel part_id negativo
        negative_key = random.choice(self.pid_to_keys[negative_pid])[0]
        negative_img = self.key_to_image[negative_key]
        
        return anchor_img, positive_img, negative_img

# La funzione per ottenere il dataloader non cambia
def get_triplet_dataloader(tar_pattern, part_id_map_path, batch_size, num_workers):
    dataset = TripletDataset(tar_pattern, part_id_map_path)
    # shuffle=True è importante per la casualità delle triplette
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)