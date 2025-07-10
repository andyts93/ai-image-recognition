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

def get_images_by_part_id(part_id):
    group = dataset_df[dataset_df['part_id'] == int(part_id)]
    return group['image_path'].tolist()

def show_next():
    if index['value'] >= len(df):
        return None, "<div style='color:green;'>✔️ Fine del dataset</div>", "", [], ""

    row = df.iloc[index['value']]
    image_path, part_id, category_id = row

    category, result = main(image_path, EMBEDDING_MODEL_PATH, MODEL_PATH, NUM_CLASSES)
    category_pred = get_category(category)

    # INFO testo
    info = f"<b>📄 File:</b> {image_path}<br>"
    info += f"<b>🆔 Part ID:</b> {part_id}<br>"
    info += f"<b>✅ Categoria Predetta:</b> {category_pred} ({category})<br>"
    info += f"<b>🎯 Categoria Corretta:</b> {category_id}<br><br>"
    info += "<b>🔍 Top risultati:</b><br>"
    for ids, score in result:
        info += f" - ID: {ids}, Score: {score:.4f}<br>"

    # Esito
    if int(category_pred) == int(category_id):
        verdict = "<div style='color:green; font-size: 20px;'>✅ Categoria corretta!</div>"
    else:
        verdict = "<div style='color:red; font-size: 20px;'>❌ Categoria sbagliata!</div>"

    try:
        image = Image.open(image_path)
    except Exception as e:
        return None, f"<div style='color:red;'>Errore: {e}</div>", "", [], ""

    # Recupera immagini simili
    similar_images = []
    if result:
        top_part_id = result[0][0]
        paths = get_images_by_part_id(top_part_id)
        for p in paths:
            try:
                img = Image.open(p)
                similar_images.append(img)
            except:
                continue

    index['value'] += 1
    return image, verdict, info, similar_images, f"{index['value']} / {len(df)}"

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## 🔍 Visualizzatore Risultati Classificazione")

        with gr.Row():
            image_output = gr.Image(label="🖼️ Immagine in esame")
            
            with gr.Column():
                verdict_output = gr.HTML(label="Esito")
                info_output = gr.HTML(label="📋 Informazioni dettagliate")

                gallery_output = gr.Gallery(label="🔁 Immagini simili dal DB", columns=(4,), height="auto")

        with gr.Column():
            count_output = gr.Textbox(label="Progresso", interactive=False)

            next_button = gr.Button("➡️ Avanti")

        next_button.click(fn=show_next, outputs=[
            image_output,
            verdict_output,
            info_output,
            gallery_output,
            count_output
        ])

    demo.launch()
