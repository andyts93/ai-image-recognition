import torch
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from config import * # Importa le tue configurazioni
from model.embedding_model import EmbeddingNet
# Importa il dataloader di validazione/inferenza (senza augmentation)
from dataset.loader import get_val_dataloader 

def calculate_all_centroids(model, dataloader, device):
    """
    Scansiona un dataset per calcolare l'embedding medio (centroide) per ogni categoria.
    """
    # Dizionario per accumulare gli embedding per ogni categoria
    embeddings_by_cat = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        # Itera su tutto il dataset di training o validazione
        for images, labels in tqdm(dataloader, desc="Estrazione embeddings per centroidi"):
            images = images.to(device)
            # Calcola gli embeddings
            embeddings = model(images).cpu() 
            
            # Raggruppa gli embeddings per la loro etichetta (category_id)
            for emb, label in zip(embeddings, labels):
                embeddings_by_cat[label.item()].append(emb)

    # Dizionario per salvare i centroidi finali
    centroids = {}
    print("\nCalcolo dei centroidi medi...")
    for cat_id, embs in tqdm(embeddings_by_cat.items()):
        # Sovrapponi tutti gli embedding di una categoria e calcola la media
        all_embs_tensor = torch.stack(embs)
        centroids[cat_id] = all_embs_tensor.mean(dim=0).numpy() # Calcola la media e converte in numpy

    return centroids

if __name__ == "__main__":
    # Carica il tuo modello di embedding già addestrato
    emb_model = EmbeddingNet()
    emb_model.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location=DEVICE))
    emb_model.to(DEVICE)

    # Usa un dataloader senza shuffling e senza augmentation per questo processo
    # Assicurati che il tuo get_val_dataloader restituisca (immagini, etichette)
    dataloader = get_val_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS, shuffle=False)
    
    # Esegui il calcolo
    category_centroids = calculate_all_centroids(emb_model, dataloader, DEVICE)
    
    # Salva i centroidi su un file per usarli nell'app
    output_path = "data/dataset/category_centroids.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(category_centroids, f)
        
    print(f"\nCentroidi salvati con successo in: {output_path}")
    print(f"Calcolati {len(category_centroids)} centroidi.")