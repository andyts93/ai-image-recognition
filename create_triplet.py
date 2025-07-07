import os
import pickle
import webdataset as wds
from collections import defaultdict
from tqdm import tqdm
from config import *

OUTPUT_PATH = "data/dataset/part_id_map.pkl"

part_id_map = defaultdict(list)

dataset = (
    wds.WebDataset(TRAIN_SHARDS, handler=wds.warn_and_continue)
    .to_tuple("pid", "__key__")
)

print("Estrazione part_id → __key__...")
for pid_bytes, key in tqdm(dataset):
    part_id = pid_bytes.decode("utf-8").strip()
    part_id_map[part_id].append(key)

print(f"Totale part_id unici: {len(part_id_map)}")

# Salva su disco
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(dict(part_id_map), f)

print(f"Salvato in: {OUTPUT_PATH}")
