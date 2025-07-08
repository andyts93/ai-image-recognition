from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from dataset.loader import get_dataloader
from config import *
import torch

if __name__ == '__main__':
    dl = get_dataloader(TRAIN_SHARDS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    for batch in dl:
        images, labels = batch
        if isinstance(images, list):
            images = torch.stack(images)
        grid = make_grid(images)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(labels)
        plt.show()
        break