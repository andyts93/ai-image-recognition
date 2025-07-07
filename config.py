import torch

CSV_PATH = "data/metadata_clean.csv"
IMG_ROOT = "."
NUM_CLASSES = 62
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SHARDS = "data/dataset/shard-{0000..0000}.tar"
VAL_SHARDS = "data/dataset/shard-{0000..0000}.tar"
PART_ID_MAP = "data/dataset/part_id_map.pkl"

MODEL_PATH = "models/classifier.pth"
EMBEDDING_MODEL_PATH = "models/embedding.pth"