import os
import csv
import pymysql
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from config import *

SOURCE_IMAGE_FOLDER = "/home/foto_cartellini/"

# === DATABASE CONNECTION ===
conn = pymysql.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    cursorclass=pymysql.cursors.DictCursor
)

if __name__ == "__main__":
    # === CSV WRITER ===
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    csv_file = open(CSV_PATH, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(['image_path', 'part_id', 'category_id'])

    try:
        with conn.cursor() as cursor:
            print("Fetching IDs...")
            cursor.execute("""
                SELECT id 
                FROM corsmagquattro.ricambi 
                WHERE visible = 1 AND id > 0 
                LIMIT 25
            """)
            ids = cursor.fetchall()

            for arow in ids:
                print(f"Processing ID: {arow['id']}")

                query = f"""
                    SELECT 
                        REPLACE(REPLACE(percorso, '--192.168.0.148-imgcartellini-', ''), '-', '/') AS image_path,
                        b.idmag AS part_id,
                        c.ania AS category_id
                    FROM corsmagquattro.foto_cartellini a
                    JOIN cors_optimized.tag b ON a.cartellino = b.cartellino
                    JOIN cors_optimized.component c ON b.idmag = c.idmag AND b.idver = c.idver
                    JOIN corsmagquattro.ricambi d ON c.ania = d.id
                    WHERE d.id = {arow['id']}
                    LIMIT 300
                """
                cursor.execute(query)
                rows = cursor.fetchall()

                if not rows:
                    print("No results found.")
                    continue

                for i, row in enumerate(tqdm(rows, desc=f"ID {arow['id']}", unit="img")):
                    image_path = row['image_path']
                    file_path = os.path.join(IMAGE_OUTPUT_FOLDER, image_path)

                    if not os.path.exists(file_path):
                        source_path = os.path.join(SOURCE_IMAGE_FOLDER, image_path)
                        if not os.path.exists(source_path):
                            print(f"Image not found: {source_path}")
                            continue
                        # # Download
                        # try:
                        #     response = requests.get(IMAGE_BASE_URL + image_path, timeout=10)
                        #     response.raise_for_status()
                        # except Exception as e:
                        #     print(f"Failed to download image: {image_path} - {e}")
                        #     continue

                        # Create directories
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)

                        try:
                            img = Image.open(source_path)
                            img = img.convert("RGB")
                            img = img.resize(IMAGE_SIZE)
                            img.save(file_path, format='JPEG', quality=100)
                        except Exception as e:
                            print(f"Failed to resize image: {image_path} - {e}")
                            continue

                    # Save row
                    writer.writerow([file_path, row['part_id'], row['category_id']])

                print(f"Ania {arow['id']} scritto.")

    finally:
        csv_file.close()
        conn.close()