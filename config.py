import torch
import glob
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT'))
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME = os.getenv('DB_NAME')

CSV_PATH = "data/metadata.csv"
TEST_CSV_PATH = "data/test.csv"
IMAGE_OUTPUT_FOLDER = "image_resized/"
IMAGE_TEST_OUTPUT_FOLDER = "image_test/"
IMAGE_BASE_URL = "http://91.187.214.100:8380/images/foto/"
IMAGE_SIZE = (224, 224)
PKL_PATH = "data/dataset/part_id_map.pkl"

IMG_ROOT = "."
NUM_CLASSES = 25
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LEARNING_RATE = 0.001215 
CLASS_UNFROZEN_LR_FACTOR = 20.0
CLASS_WEIGHT_DECAY = 0.001092

TRAIN_SHARDS = sorted(glob.glob("data/dataset/train/shard-*.tar"))
VAL_SHARDS = sorted(glob.glob("data/dataset/val/shard-*.tar"))

PART_ID_MAP = "data/dataset/part_id_map.pkl"

MODEL_PATH = "models/classifier.pth"
EMBEDDING_MODEL_PATH = "models/embedding.pth"
FAISS_INDICES = "faiss_indices"

ALPHA = 0.5
BETA = 0.3
GAMMA = 0.2
