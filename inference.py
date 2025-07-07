from PIL import Image
import torch
from torchvision import transforms
from model.classifier import get_model
from config import *
import json

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval().to(DEVICE)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()

    with open("data/dataset/cat2idx.json", "r") as f:
        cat2idx = json.load(f)
        idx2cat = {v: k for k, v in cat2idx.items()}

    category_id = int(idx2cat[int(pred)])

    print(f"Predicted category: {category_id}")
    return pred

if __name__ == "__main__":
    import sys
    predict(sys.argv[1])