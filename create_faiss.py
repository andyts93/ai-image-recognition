import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import faiss
import os
import pickle
from collections import defaultdict
import webdataset as wds
from torchvision import transforms
from dataset.inference_loader import get_inference_dataloader
from model.embedding_model import EmbeddingNet
from config import *

os.makedirs(FAISS_INDICES, exist_ok=True)

if __name__ == "__main__":
    dataloader = get_inference_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS)

    model = EmbeddingNet(embedding_dim=128)
    model.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    embeddings = []
    part_ids = []
    category_ids = []

    with torch.no_grad():
        for images, categories, pids in tqdm(dataloader, desc="Extracting embeddings"):
            img = images.to(DEVICE)
            emb = model(img).cpu().numpy()

            embeddings.append(emb)
            part_ids.extend([str(p.item()) if isinstance(p, torch.Tensor) else str(p) for p in pids])
            category_ids.extend([int(c.item()) if isinstance(c, torch.Tensor) else int(c) for c in categories])

    embeddings = np.vstack(embeddings)

    cat_to_embeddings = defaultdict(list)
    cat_to_metadata = defaultdict(list)

    for emb, pid, cat_id in zip(embeddings, part_ids, category_ids):
        cat_to_embeddings[cat_id].append(emb)
        cat_to_metadata[cat_id].append({"part_id": pid})

    for cat_id, emb_list in cat_to_embeddings.items():
        xb = np.vstack(emb_list).astype("float32")
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(xb)

        faiss.write_index(index, os.path.join(FAISS_INDICES, f"index_{cat_id}.index"))
        with open(os.path.join(FAISS_INDICES, f"metadata_{cat_id}.npy"), "wb") as f:
            np.save(f, cat_to_metadata[cat_id])
        
        print(f"Saved FAISS index and metadata for category {cat_id} ({len(emb_list)} items)")