import torch
from tqdm import tqdm
from config import *
from dataset.loader import get_dataloader
from model.classifier import get_model
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import json

def evaluate(model):
    loader = get_dataloader(VAL_SHARDS, BATCH_SIZE, NUM_WORKERS, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

def train_model():
    dataloader = get_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS)

    for x, y in dataloader:
        print("Batch label distribution:", torch.bincount(y))
        break
    
    # model = get_model(NUM_CLASSES).to(DEVICE)
    # optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # with open("data/dataset/train_class_counts.json") as f:
    #     class_counts = json.load(f)
    
    # counts_tensor = torch.tensor([class_counts.get(str(i), 0) for i in range(NUM_CLASSES)], dtype=torch.float)
    # weights = 1.0 / counts_tensor
    # weights = weights / weights.sum() * len(class_counts)

    # criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    # num_training_steps = MAX_BATCH_PER_EPOCH * NUM_EPOCHS
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)
    # # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='max')

    # best_val_acc = 0.0

    # for epoch in range(NUM_EPOCHS):
    #     model.train()
    #     running_loss, total = 0.0, 0

    #     pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    #     for images, labels in pbar:
    #         images, labels = images.to(DEVICE), labels.to(DEVICE)

    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item() * images.size(0)
    #         pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * BATCH_SIZE))
    #         total += 1

    #     val_acc = evaluate(model)

    #     tqdm.write(f"Epoch {epoch+1} Loss: {running_loss/total:.4f} | Acc: {val_acc:.4f}")

    #     scheduler.step(val_acc)

    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), MODEL_PATH)
    #         print(f"Modello salvato in {MODEL_PATH} with vall acc: {best_val_acc:.4f}")

    # print("Training complete!")
    # print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()