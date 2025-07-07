import torch
from dataset.loader import get_dataloader
from model.classifier import get_model
from config import *

def evaluate():
    dataloader = get_dataloader(VAL_SHARDS, BATCH_SIZE, NUM_WORKERS, shuffle=False)
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval().to(DEVICE)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    evaluate()