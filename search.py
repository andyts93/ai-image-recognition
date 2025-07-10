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

def predict_category(img_tensor, cls_model):
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        logits = cls_model(img_tensor)
        predicted = torch.argmax(logits, dim=1).item()
        return predicted

def extract_embedding(img_tensor, emb_model):
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        embedding = emb_model(img_tensor).cpu().numpy()
        return embedding.astype("float32")

def query_faiss(embedding, category_id):
    index_path = f"{FAISS_INDICES}/index_{category_id}.index"
    meta_path = f"{FAISS_INDICES}/metadata_{category_id}.npy"

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Indice per categoria {category_id} non trovato")

    index = faiss.read_index(index_path)
    metadata = np.load(meta_path, allow_pickle=True)

    D, I = index.search(embedding, k=50)
    results = [metadata[i] for i in I[0]]
    distances = D[0]
    
    # Aggrega per part_id
    agg = defaultdict(list)
    for res, dist in zip(results, distances):
        part_id = res['part_id']
        agg[part_id].append(dist)

    scored_part_ids = []
    for part_id, dists in agg.items():
        dists = np.array(dists)
        mean_d = dists.mean()
        min_d = dists.min()
        var_d = dists.var()

        score = ALPHA * mean_d + BETA * min_d + GAMMA * var_d
        scored_part_ids.append((part_id, score))

    # Ordina per score crescente
    scored_part_ids.sort(key=lambda x: x[1])

    return scored_part_ids[:5]

def main(img_path, embedding_path, classifier_path, num_classes):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)

    emb_model, cls_model = load_models(embedding_path, classifier_path, num_classes)

    category_id = predict_category(img_tensor, cls_model)

    embedding = extract_embedding(img_tensor, emb_model)
    return category_id, query_faiss(embedding, category_id)
    # results, distances = query_faiss(embedding, category_id)

    # print("\nRisultati trovati")
    # for i, (res, dist) in enumerate(zip(results, distances)):
    #     print(f"{i+1}. Part ID: {res['part_id']}, Distance: {dist:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Percorso dell'immagine da cercare")
    
    args = parser.parse_args()

    main(args.image_path, EMBEDDING_MODEL_PATH, MODEL_PATH, NUM_CLASSES)