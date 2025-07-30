""" Script per estrarre i part id per le prove con Pollini """
import pandas as pd
from config import *
import pymysql
import csv
from tqdm import tqdm

df = pd.read_csv(CSV_PATH)
unique_part_ids = df['part_id'].unique()

# === DATABASE CONNECTION ===
conn = pymysql.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with conn.cursor() as cursor:
        csv_file = open("components.csv", mode="w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(['Marca', 'Modello', 'Descrizione'])

        for comp_id in tqdm(unique_part_ids, unit="component"):
            cursor.execute(f"SELECT dsMar, dsMod, descr FROM cors_optimized.component WHERE idmag = {comp_id}")
            component = cursor.fetchone()

            if component:
                writer.writerow([component["dsMar"], component["dsMod"], component["descr"]])
finally:
    csv_file.close()
