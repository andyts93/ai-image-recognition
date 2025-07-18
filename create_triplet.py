import os
import pickle
import webdataset as wds
from collections import defaultdict
from tqdm import tqdm
from config import *

part_id_map = defaultdict(list)

dataset = (
    wds.WebDataset(TRAIN_SHARDS, handler=wds.warn_and_continue)
    .to_tuple("pid", "__key__", "cls")
)

print("Estrazione part_id → __key__...")
for pid_bytes, key, cls in tqdm(dataset):
    part_id = pid_bytes.decode("utf-8").strip()
    part_id_map[part_id].append((key, cls))

print(f"Totale part_id unici: {len(part_id_map)}")

# Salva su disco
with open(PKL_PATH, "wb") as f:
    pickle.dump(dict(part_id_map), f)

print(f"Salvato in: {PKL_PATH}")
