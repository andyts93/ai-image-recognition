import os
import pandas as pd
import json
import json
import webdataset as wds
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import pickle
import io

CSV_PATH = "data/metadata.csv"
OUTPUT_DIR = "data/dataset"
SHARD_SIZE = 100
IMAGE_ROOT = "."
OUTPUT_PATH = "data/dataset/part_id_map.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["image_path", "category_id", "part_id"])
df["image_path"] = df["image_path"].apply(lambda p: os.path.join(IMAGE_ROOT, p))

unique_categories = sorted(df["category_id"].unique())
cat2idx = {str(cid): idx for idx, cid in enumerate(unique_categories)}

with open(os.path.join(OUTPUT_DIR, 'cat2idx.json'), 'w') as f:
    json.dump(cat2idx, f)

shard_id = 0
sample_id = 0
sink = None
part_id_map = defaultdict(list)

def start_new_sink(shard_id):
    tarpath = os.path.join(OUTPUT_DIR, f"shard-{shard_id:04d}.tar")
    return wds.TarWriter(tarpath)

if __name__ == "__main__":
    sink = start_new_sink(shard_id)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            image_path = row["image_path"]
            category_id = str(row["category_id"])
            part_id = str(row["part_id"])
            cls_index = cat2idx[category_id]

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                img_bytes = buffer.getvalue()
            
            sample = {
                "__key__": f"{sample_id:08d}",
                "jpg": img_bytes,
                "cls": str(cls_index).encode("utf-8"),
                "pid": part_id.encode("utf-8"),
            }
            part_id_map[part_id].append(sample_id)

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

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(dict(part_id_map), f)

    print(f"\nCompletato: {sample_id} immagini scritte in {shard_id + 1} shards")