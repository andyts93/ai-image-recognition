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
                WHERE id IN (33,34,38,41,47,50,51,53,57,66,68,70,74,76,80,84,86,87,92,97,98,106,146,148,153,156,163,164,166,193,199,200,225,228,231,234,237,245,274,287,316,317,318,325,329,336,337,341,342,364,376,382,384,387,388,444,454,456,458,474,475,506,512,520,528,536,539,556,561,564,576,585,638,677,681,692,697,710,711,715,728,819,825,828,868,874,875,878,881,883,884,886,963,982,986,990,991,996,999,1000,1001,1064,1100,1112,1118,1122,1130,1133,1135,1143,1244,1343,1344,1373,1383,1385,1418,1439,1443,1447,1457,1458,1460,1464,1474,1488,1492,1519,1563,1565,1586,1591,1606,1607,1635,1694,1771,1774,1776,1777,1779,1791,1793,1797,1798,1799,1800,1817,1876,1891,1918,1919,1935,1945,1966,1972,1981,1982,1992,1993,1995,1999,2000,2067,2069,2090,2091,2096,2099,2100,2136,2138,2140,2141,2144,2150,2160,2162,2168,2175,2177,2183,2187,2191,2200,2209,2216,2275,2295,2300,2301,2302,2306,2308,2309,2319,2350,2383,2392,2393,2411,2413,2414,2467,2470,2489,2499,2558,2561,2566,2580,2643,2651,2662,2707,2721,2731,2739,2742,2743,2745,2746,2748,2766,2767,2768,2770,2772,2783)
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

                    if not os.path.exists(target_path):
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
