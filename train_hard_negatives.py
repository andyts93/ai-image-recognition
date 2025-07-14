import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.embedding_model import EmbeddingNet
from config import *
import torch.nn.functional as F
import numpy as np

# Importa entrambi i dataloader
from dataset.part_loader import get_part_dataloader, get_val_dataloader

# La funzione per il mining rimane la stessa
def mine_hard_triplets(embeddings, pids):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    is_pos = pids.unsqueeze(1) == pids.unsqueeze(0)
    is_neg = ~is_pos
    dist_pos = dist_matrix.clone()
    dist_pos[is_neg] = -float('inf')
    hardest_positive_idx = dist_pos.argmax(dim=1)
    dist_neg = dist_matrix.clone()
    dist_neg[is_pos] = float('inf')
    hardest_negative_idx = dist_neg.argmin(dim=1)
    return embeddings, embeddings[hardest_positive_idx], embeddings[hardest_negative_idx]

def evaluate(model, val_loader, device, k=5):
    """ Calcola il Recall@K sul set di validazione. """
    model.eval()
    all_embeddings = []
    all_pids = []
    
    # 1. Estrai tutti gli embedding e i pids dal set di validazione
    with torch.no_grad():
        for images, pids in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            embeddings = F.normalize(model(images), p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            all_pids.extend(pids.cpu().numpy())

    all_embeddings = torch.cat(all_embeddings)
    all_pids = torch.from_numpy(np.array(all_pids))

    # Calcola la matrice delle distanze completa
    dist_matrix = torch.cdist(all_embeddings, all_embeddings, p=2)
    
    # Trova i K vicini più prossimi per ogni elemento (escludendo se stesso)
    # Aggiungiamo un valore alto alla diagonale per escludere l'identità
    dist_matrix.fill_diagonal_(float('inf'))
    _, top_k_indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)

    # Prendi i pids dei top K vicini
    top_k_pids = all_pids[top_k_indices]
    
    # Prepara i pids originali per il confronto
    anchor_pids = all_pids.unsqueeze(1).expand_as(top_k_pids)
    
    # Calcola dove c'è una corrispondenza
    correct_matches = torch.any(top_k_pids == anchor_pids, dim=1)
    
    # Calcola il recall
    recall_at_k = correct_matches.float().mean().item()
    
    return recall_at_k


def train_embedding_model(epochs=10, lr=1e-4, embedding_dim=128, margin=0.5, device=DEVICE):
    # 2. Prepara entrambi i dataloader
    train_loader = get_part_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS)
    val_loader = get_val_dataloader(VAL_SHARDS, BATCH_SIZE, NUM_WORKERS)
    
    model = EmbeddingNet(embedding_dim=embedding_dim).to(device)
    criterion = nn.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5) # Modalità 'max' per il recall

    best_val_recall = 0.0

    print("Inizio addestramento con Online Hard Negative Mining...")
    for epoch in range(1, epochs + 1):
        model.train() # Imposta il modello in modalità training
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, pids in pbar:
            images, pids = images.to(device), pids.to(device)
            optimizer.zero_grad()
            embeddings = F.normalize(model(images), p=2, dim=1)
            anchor_embs, positive_embs, negative_embs = mine_hard_triplets(embeddings, pids)
            loss = criterion(anchor_embs, positive_embs, negative_embs)
            if loss > 0:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()

            num_batches += 1
            pbar.set_postfix(loss=loss.item())

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
        else:
            avg_loss = 0.0
        
        # 3. Esegui la valutazione alla fine di ogni epoca
        val_recall_at_5 = evaluate(model, val_loader, device, k=5)
        
        tqdm.write(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f} | Validation Recall@5: {val_recall_at_5:.4f}")
        
        # Aggiorna lo scheduler in base al recall
        scheduler.step(val_recall_at_5)

        # 4. Salva il modello se le performance di validazione sono migliorate
        if val_recall_at_5 > best_val_recall:
            best_val_recall = val_recall_at_5
            save_path = os.path.join(MODELS_DIR, "best_embedding_model.pth")
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"🚀 Nuovo modello migliore salvato in {save_path} con Recall@5: {best_val_recall:.4f}")

    print("Addestramento completato.")
    print(f"Miglior Recall@5 di validazione ottenuto: {best_val_recall:.4f}")
    return model

if __name__ == "__main__":
    train_embedding_model(epochs=NUM_EPOCHS)