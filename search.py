import torch
import torchvision.transforms as transforms
from PIL import Image
import faiss
import numpy as np
import json
import os
import argparse
from config import *
from collections import defaultdict
import torch.nn.functional as F
import pickle

from model.embedding_model import EmbeddingNet
from model.classifier import get_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
with open("data/dataset/category_centroids.pkl", "rb") as f:
    category_centroids = pickle.load(f)

# Definisci una soglia di distanza (da trovare sperimentalmente)
DISTANCE_THRESHOLD = 20.0

def load_models(embedding_path, classifier_path, num_classes):
    emb_model = EmbeddingNet()
    emb_model.load_state_dict(torch.load(embedding_path, map_location=DEVICE))
    emb_model.to(DEVICE)
    emb_model.eval()

    cls_model = get_model(num_classes=num_classes)
    cls_model.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
    cls_model.to(DEVICE)
    cls_model.eval()

    return emb_model, cls_model

def predict_category(img_tensor, cls_model, k=3):
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        logits = cls_model(img_tensor)
        
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=1)
        
        # Converte in lista di tuple: (classe, probabilità)
        results = [
            (topk_indices[0][i].item(), topk_probs[0][i].item())
            for i in range(k)
        ]
        return results
        
def extract_embedding(img_tensor, emb_model):
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        embedding = emb_model(img_tensor).cpu().numpy()
        return embedding.astype("float32")

def query_faiss(embedding, category_id, faiss_k, alpha, beta, gamma):
    index_path = f"{FAISS_INDICES}/index_{category_id}.index"
    meta_path = f"{FAISS_INDICES}/metadata_{category_id}.npy"

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Indice per categoria {category_id} non trovato")

    index = faiss.read_index(index_path)
    metadata = np.load(meta_path, allow_pickle=True)

    D, I = index.search(embedding, k=faiss_k)
    results = [metadata[i] for i in I[0]]
    distances = D[0]
    
    # Aggrega per part_id
    agg = defaultdict(list)
    for res, dist in zip(results, distances):
        part_id = res['part_id']
        agg[part_id].append(dist)

    print(agg)

    scored_part_ids = []
    for part_id, dists in agg.items():
        dists = np.array(dists)
        mean_d = dists.mean()
        min_d = dists.min()
        var_d = dists.var()

        score = alpha * mean_d + beta * min_d + gamma * var_d
        scored_part_ids.append((part_id, score))

    # Ordina per score crescente
    scored_part_ids.sort(key=lambda x: x[1])

    return scored_part_ids[:3]

def weighted_avg(dists):
    weights = 1 / (np.arange(1, len(dists)+1))  # [1, 0.5, 0.33, ...]
    return np.average(sorted(dists), weights=weights[:len(dists)])

def main(img, embedding_path, classifier_path, num_classes, params):
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")
    img_tensor = transform(img)

    emb_model, cls_model = load_models(embedding_path, classifier_path, num_classes)

    embedding = extract_embedding(img_tensor, emb_model)

    categories = predict_category(img_tensor, cls_model, k=params["top_k_classifier"])
    print(f"Categories: {categories}")

    # for predicted_cat_id, confidence in categories:
    #     if predicted_cat_id in category_centroids:
    #         # Recupera il centroide per la categoria predetta
    #         centroid = category_centroids[predicted_cat_id]
    #
    #         # Calcola la distanza Euclidea (L2) tra l'embedding e il centroide
    #         distance = np.linalg.norm(embedding - centroid)
    #
    #         print(f"Categoria predetta: {predicted_cat_id}, Distanza dal centroide: {distance:.4f}")
    #
    #         # Se la distanza supera la soglia, è un'anomalia
    #         if distance > DISTANCE_THRESHOLD:
    #             # Puoi personalizzare l'output per l'utente
    #             print(f"Immagine non riconosciuta (troppo diversa dai ricambi noti)")
    #     else:
    #         # Se non abbiamo un centroide per questa categoria, non possiamo verificare
    #         print(f"Attenzione: Centroide non trovato per la categoria {predicted_cat_id}")

    dist_by_cat = defaultdict(list)
    all_results = []

    for cat_id, prob in categories:
        if prob > params['prob_threshold']:
            results = query_faiss(embedding, cat_id, 
                                faiss_k=params['faiss_k'], 
                                alpha=params['alpha'], 
                                beta=params['beta'], 
                                gamma=params['gamma'])
            for part_id, distance in results:
                dist = float(distance)
                all_results.append((cat_id, part_id, dist))
                dist_by_cat[cat_id].append(dist)

    # Calcola media distanza per categoria
    avg_dist_by_cat = {
        cat_id: weighted_avg(dists)
        for cat_id, dists in dist_by_cat.items()
    }

    prob_by_cat = {cat_id: prob for cat_id, prob in categories}

    # 1. Crea una lista arricchita con lo score globale calcolato per ogni risultato
    results_with_scores = []
    for cat_id, part_id, part_score in all_results:
        # Prende lo score medio della categoria e la sua probabilità
        category_avg_score = avg_dist_by_cat.get(cat_id, float('inf'))
        category_prob = prob_by_cat.get(cat_id, 1e-9) # Usa un valore piccolo per evitare divisione per zero

        # Calcola lo score globale, che è la metrica primaria per l'ordinamento
        global_score = category_avg_score / category_prob
        
        results_with_scores.append({
            'part_id': part_id,
            'global_score': global_score,
            'cat_id': cat_id,
            'part_score': part_score # Manteniamo lo score parziale per l'ordinamento secondario
        })

    # 2. Ordina la lista usando gli score pre-calcolati
    sorted_results = sorted(
        results_with_scores,
        key=lambda x: (x['global_score'], x['part_score'])
    )

    # 3. Crea l'output finale usando lo score globale ("global_score")
    top_results = [
        (item['part_id'], item['global_score'], item['cat_id'], item['part_score']) 
        for item in sorted_results
    ]

    print(top_results)

    return top_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Percorso dell'immagine da cercare")
    
    args = parser.parse_args()

    print(main(args.image_path, EMBEDDING_MODEL_PATH, MODEL_PATH, NUM_CLASSES))