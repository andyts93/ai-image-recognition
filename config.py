import torch
import glob
import random

CSV_PATH = "data/metadata_clean.csv"
IMG_ROOT = "."
NUM_CLASSES = 62
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_SHARDS = sorted(glob.glob("data/dataset/shard-*.tar"))
random.seed(42)
random.shuffle(ALL_SHARDS)
split_index = int(0.8 * len(ALL_SHARDS))
TRAIN_SHARDS = ALL_SHARDS[:split_index]
VAL_SHARDS = ALL_SHARDS[split_index:]
MAX_BATCH_PER_EPOCH = 503 / BATCH_SIZE

PART_ID_MAP = "data/dataset/part_id_map.pkl"

MODEL_PATH = "models/classifier.pth"
EMBEDDING_MODEL_PATH = "models/embedding.pth"
FAISS_INDICES = "faiss_indices"