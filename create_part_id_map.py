import pandas as pd
import json
from config import CSV_PATH  # Assicurati che punti al tuo CSV con i metadati

print("Inizio la creazione della mappa part_id -> idx...")

# Carica solo le colonne necessarie
df = pd.read_csv(CSV_PATH, usecols=['part_id'], dtype={'part_id': str})
df = df.dropna(subset=['part_id'])

# Trova tutti i part_id unici e ordinali
unique_part_ids = sorted(df['part_id'].unique())

# Crea la mappa: {part_id_originale: indice_nuovo}
part_id_to_idx = {part_id: i for i, part_id in enumerate(unique_part_ids)}

# Salva la mappa in un file JSON
output_path = "data/dataset/part_id_to_idx.json"
with open(output_path, 'w') as f:
    json.dump(part_id_to_idx, f)

num_unique = len(unique_part_ids)
print(f"Mappa creata e salvata in: {output_path}")
print(f"Numero totale di part_id unici: {num_unique}")

# Salva anche il numero totale per caricarlo facilmente nello script di training
with open("data/dataset/num_unique_part_ids.txt", "w") as f:
    f.write(str(num_unique))