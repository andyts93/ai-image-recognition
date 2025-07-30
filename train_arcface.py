import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from config import *  # Le tue configurazioni
# Riusiamo il part_loader perché ci dà (immagine, etichetta)
from dataset.part_loader import get_part_dataloader, get_val_dataloader
from model.embedding_model import EmbeddingNetArcFace  # Il nuovo modello
# Riusiamo la funzione di valutazione perché funziona con gli embedding
from train_hard_negatives import evaluate


def train_arcface_model(epochs=30, lr=1e-4, embedding_dim=128, device=DEVICE):
    # Prepara i dataloader
    # NUM_CLASSES è il numero di part_id unici, non di categorie!
    # Assicurati che il tuo dataloader usi i part_id come etichette
    train_loader = get_part_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS)
    val_loader = get_val_dataloader(VAL_SHARDS, BATCH_SIZE, NUM_WORKERS)

    # Inizializza il nuovo modello
    model = EmbeddingNetArcFace(embedding_dim=embedding_dim, num_classes=NUM_UNIQUE_PART_IDS).to(device)

    # La loss è una semplice CrossEntropyLoss!
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_recall = 0.0

    print("Inizio addestramento con ArcFace Loss...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, pids in pbar:
            images, pids = images.to(device), pids.to(device)

            optimizer.zero_grad()

            # Passa sia le immagini che le etichette (pids) al modello
            outputs = model(images, label=pids)

            loss = criterion(outputs, pids)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
        else:
            avg_loss = 0.0

        # La funzione 'evaluate' funziona perché il modello, se chiamato senza 'label',
        # restituisce solo gli embedding, proprio come si aspetta la funzione.
        val_recall_at_5 = evaluate(model, val_loader, device, k=5)

        tqdm.write(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f} | Validation Recall@5: {val_recall_at_5:.4f}")

        if val_recall_at_5 > best_val_recall:
            best_val_recall = val_recall_at_5
            save_path = os.path.join("models/", "best_embedding_model_arcface.pth")
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"🚀 Nuovo modello migliore salvato in {save_path} con Recall@5: {best_val_recall:.4f}")

    print(f"Miglior Recall@5 di validazione ottenuto: {best_val_recall:.4f}")
    return model


if __name__ == "__main__":
    # IMPORTANTE: Devi conoscere il numero di part_id unici nel tuo dataset
    # Questo valore è cruciale per la dimensione del layer di classificazione di ArcFace
    NUM_UNIQUE_PART_IDS = 92526  # Sostituisci con il tuo valore reale
    train_arcface_model(epochs=NUM_EPOCHS)