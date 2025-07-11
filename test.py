import gradio as gr
import pandas as pd
from PIL import Image
import json
from config import *
from search import main

# Carica mapping categorie
with open('data/dataset/cat2idx.json') as f:
    cat2idx = json.load(f)

def get_category(idx):
    return next((k for k, v in cat2idx.items() if v == idx), None)

df = pd.read_csv(TEST_CSV_PATH)
index = {'value': 0}
dataset_df = pd.read_csv(CSV_PATH)
dataset_df['part_id'] = dataset_df['part_id'].astype(str)

def get_images_by_part_id(part_id):
    part_id = str(part_id)
    group = dataset_df[dataset_df['part_id'] == part_id]
    return group['image_path'].tolist()

def show_next():
    if index['value'] >= len(df):
        return None, "<div style='color:green;'>✔️ Fine del dataset</div>", "", [], ""

    row = df.iloc[index['value']]
    image_path, part_id, category_id = row
    similar_images = []

    results = main(image_path, EMBEDDING_MODEL_PATH, MODEL_PATH, NUM_CLASSES)
    info = f"<b>📄 File:</b> {image_path}<br><br><br>"
    for cat_id, part_id, score in results:
        category_pred = get_category(cat_id)

        # INFO testo
        info += f"<b>🆔 Part ID:</b> {part_id}<br>"
        info += f"<b>🔮 Categoria Predetta:</b> {category_pred} ({cat_id})<br>"
        info += f"<b>🎯 Categoria Corretta:</b> {category_id}<br>"
        if int(category_pred) == int(category_id):
            info += f"<b>CORRETTA</b><br>"
        info += f"<b>Score</b>: {score:.4f}<br><br>"

        paths = get_images_by_part_id(part_id)
        for p in paths:
            try:
                img = Image.open(p)
                similar_images.append((img, f"P: {part_id}|C:{category_pred}"))
            except:
                continue

    try:
        image = Image.open(image_path)
    except Exception as e:
        return None, f"<div style='color:red;'>Errore: {e}</div>", "", [], ""

    index['value'] += 10
    return image, info, similar_images, f"{index['value']} / {len(df)}"

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## 🔍 Visualizzatore Risultati Classificazione")

        with gr.Row():
            image_output = gr.Image(label="🖼️ Immagine in esame")
            
            with gr.Column():
                info_output = gr.HTML(label="📋 Informazioni dettagliate")

                gallery_output = gr.Gallery(label="🔁 Immagini simili dal DB", columns=(4,), height="auto", show_label=True)

        with gr.Column():
            count_output = gr.Textbox(label="Progresso", interactive=False)

            next_button = gr.Button("➡️ Avanti")

        next_button.click(fn=show_next, outputs=[
            image_output,
            info_output,
            gallery_output,
            count_output
        ])

    demo.launch()
