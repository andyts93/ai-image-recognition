import torch
from tqdm import tqdm
from config import *
from dataset.loader import get_dataloader
from model.classifier import get_model
import torch.nn as nn
from torch.optim import Adam

def train_model():
    dataloader = get_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS)
    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1} Loss: {running_loss/total:.4f} | Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modello salvato in {MODEL_PATH}")

if __name__ == "__main__":
    train_model()