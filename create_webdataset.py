import os
import pandas as pd
import json
import webdataset as wds
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import pickle
import io
import random
from collections import Counter
from sklearn.utils import resample

CSV_PATH = "data/metadata.csv"
OUTPUT_DIR = "data/dataset"
SHARD_SIZE = 50
IMAGE_ROOT = "."
OUTPUT_PATH = "data/dataset/part_id_map.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["image_path", "category_id", "part_id"])
df["image_path"] = df["image_path"].apply(lambda p: os.path.join(IMAGE_ROOT, p))

counts = df["category_id"].value_counts()
max_count = counts.max()

df_list = [df[df["category_id"] == cid] for cid in counts.index]

df_oversampled = []
for subset in df_list:
    n = len(subset)
    if n < max_count:
        df_up = resample(subset, replace=True, n_samples=max_count, random_state=42)
        df_oversampled.append(df_up)
    else:
        df_oversampled.append(subset)

df = pd.concat(df_oversampled).reset_index(drop=True)

unique_categories = sorted(df["category_id"].unique())
cat2idx = {str(cid): idx for idx, cid in enumerate(unique_categories)}

with open(os.path.join(OUTPUT_DIR, 'cat2idx.json'), 'w') as f:
    json.dump(cat2idx, f)

# Raggruppa i sample per part_id
partid_to_rows = defaultdict(list)
for idx, row in df.iterrows():
    partid_to_rows[row["part_id"]].append(row)

# Lista part_id da suddividere in train/val
all_part_ids = list(partid_to_rows.keys())
random.seed(42)
random.shuffle(all_part_ids)

split_idx = int(0.8 * len(all_part_ids))
train_part_ids = set(all_part_ids[:split_idx])
val_part_ids = set(all_part_ids[split_idx:])

def write_shards(part_ids, output_subdir):
    shard_id = 0
    sample_id = 0
    sink = None

    def start_new_sink(shard_id):
        tarpath = os.path.join(OUTPUT_DIR, output_subdir, f"shard-{shard_id:04d}.tar")
        return wds.TarWriter(tarpath)

    sink = start_new_sink(shard_id)
    for part_id in tqdm(part_ids, desc=f"Writing {output_subdir} shards"):
        for row in partid_to_rows[part_id]:
            try:
                image_path = row["image_path"]
                category_id = str(row["category_id"])
                cls_index = cat2idx[category_id]

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    img_bytes = buffer.getvalue()

                sample = {
                    "__key__": f"{sample_id:08d}",
                    "jpg": img_bytes,
                    "cls": int(cls_index),
                    "pid": str(part_id).encode("utf-8"),
                }

                sink.write(sample)
                sample_id += 1

                if sample_id % SHARD_SIZE == 0:
                    sink.close()
                    shard_id += 1
                    sink = start_new_sink(shard_id)

            except Exception as e:
                print(f"Errore su {row['image_path']}: {repr(e)}")
                continue
    if sink:
        sink.close()
    print(f"Completati {sample_id} samples in {shard_id+1} shards in {output_subdir}")

if __name__ == "__main__":
    write_shards(train_part_ids, "train")
    write_shards(val_part_ids, "val")