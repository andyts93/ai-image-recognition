import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

#=== CONFIG ===
METADATA_IN = "metadata.csv"
IMAGE_FOLDER_IN = "samples/"
IMAGE_FOLDER_OUT = "image_resized/"
METADATA_OUT = "metadata_clean.csv"
IMAGE_SIZE = (224, 224)

# === FUNCTIONS ===

def resize_and_save_image(in_path, out_path, size):
    try:
        with Image.open(in_path) as img:
            img = img.convert("RGB")
            img = img.resize(size)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img.save(out_path, optimize=True, quality=85)
        return True
    except Exception as e:
        print(f"Error processing {in_path}: {e}")
        return False

# === MAIN SCRIPT ===
print("Loading metadata...")
df = pd.read_csv(METADATA_IN)
clean_rows = []

print("Processing images...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    rel_path = row['image_path']
    abs_path_in = os.path.join(IMAGE_FOLDER_IN, rel_path)
    abs_path_out = os.path.join(IMAGE_FOLDER_OUT, rel_path)

    if not os.path.exists(abs_path_in):
        print(f"Image not found: {abs_path_in}")
        continue
    
    ok = resize_and_save_image(abs_path_in, abs_path_out, IMAGE_SIZE)
    if ok:
        row['image_path'] = abs_path_out
        clean_rows.append(row)

print("Saving cleaned metadata...")
df_clean = pd.DataFrame(clean_rows)
df_clean.to_csv(METADATA_OUT, index=False)

print("Done! Cleaned metadata saved to", METADATA_OUT)