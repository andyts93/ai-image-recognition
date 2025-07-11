import os
import csv
import pymysql
from PIL import Image
from tqdm import tqdm
from config import *

# === DATABASE CONNECTION ===
conn = pymysql.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    cursorclass=pymysql.cursors.DictCursor
)

SOURCE_IMAGE_FOLDER = "/mnt/foto_cartellini"

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
                    UNION
                        SELECT 
                            REPLACE(REPLACE(path, 'images/foto/', ''), '-', '/') AS image_path,
                            c.idmag AS part_id,
                            c.ania AS category_id
                        FROM cors_optimized.component_image a
                        JOIN cors_optimized.component c ON a.idmag = c.idmag
                        JOIN corsmagquattro.ricambi d ON c.ania = d.id
                        WHERE d.id = {arow['id']}
                """
                cursor.execute(query)
                rows = cursor.fetchall()

                if not rows:
                    print("No results found.")
                    continue

                for row in tqdm(rows, desc=f"ID {arow['id']}", unit="img"):
                    relative_image_path = row['image_path']
                    source_path = os.path.join(SOURCE_IMAGE_FOLDER, relative_image_path)
                    target_path = os.path.join(IMAGE_OUTPUT_FOLDER, relative_image_path)

                    if not os.path.exists(source_path):
                        print(f"Image not found: {source_path}")
                        continue

                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with Image.open(source_path) as img:
                            img = img.convert("RGB")
                            img = img.resize(IMAGE_SIZE)
                            img.save(target_path, format='JPEG', quality=100)
                    except Exception as e:
                        print(f"Error processing image {source_path}: {e}")
                        continue

                    writer.writerow([target_path, row['part_id'], row['category_id']])

                print(f"Ania {arow['id']} scritto.")

    finally:
        csv_file.close()
        conn.close()
