import torch
import glob
import random

DB_HOST = "91.187.214.100"
DB_PORT = 8306
DB_USER = "root"
DB_PASS = "marzia69"
DB_NAME = "corsmagquattro"

CSV_PATH = "data/metadata.csv"
TEST_CSV_PATH = "data/test.csv"
IMAGE_OUTPUT_FOLDER = "image_resized/"
IMAGE_TEST_OUTPUT_FOLDER = "image_test/"
IMAGE_BASE_URL = "http://91.187.214.100:8380/images/foto/"
IMAGE_SIZE = (224, 224)

IMG_ROOT = "."
NUM_CLASSES = 25
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_SHARDS = sorted(glob.glob("data/dataset/shard-*.tar"))
random.seed(42)
random.shuffle(ALL_SHARDS)
split_index = int(0.8 * len(ALL_SHARDS))
TRAIN_SHARDS = ALL_SHARDS[:split_index]
VAL_SHARDS = ALL_SHARDS[split_index:]
TOTAL_SAMPLES = 6495
MAX_BATCH_PER_EPOCH = TOTAL_SAMPLES // BATCH_SIZE

PART_ID_MAP = "data/dataset/part_id_map.pkl"

MODEL_PATH = "models/classifier.pth"
EMBEDDING_MODEL_PATH = "models/embedding.pth"
FAISS_INDICES = "faiss_indices"

ALPHA = 0.5
BETA = 0.3
GAMMA = 0.2
