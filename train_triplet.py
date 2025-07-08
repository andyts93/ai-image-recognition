import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.embedding_model import EmbeddingNet
from config import *

def train_embedding_model(
    train_loader,
    epochs=10,
    lr=1e-3,
    embedding_dim=128,
    margin=1.0,
    device=None,
    save_path="models/embedding.pth"
):
    model = EmbeddingNet(embedding_dim=embedding_dim).to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for anchor, positive, negative in pbar:
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            optimizer.zero_grad()
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / ((pbar.n + 1) * BATCH_SIZE))
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} - Avg loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"{save_path}_epoch{epoch}.pth")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

if __name__ == "__main__":
    from dataset.triplet_loader import get_triplet_dataloader
    
    train_loader = get_triplet_dataloader(TRAIN_SHARDS, PART_ID_MAP, BATCH_SIZE, NUM_WORKERS)

    train_embedding_model(
        train_loader,
        NUM_EPOCHS
    )