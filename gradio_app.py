import gradio as gr
from search import main as search_main # Rinominiamo 'main' per chiarezza
from PIL import Image
from config import *
import json
import pymysql
import pickle
import webdataset as wds
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURAZIONE E CARICAMENTO GLOBALE (eseguito una sola volta) ---

# Connessione al DB
conn = pymysql.connect(
    host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
    database=DB_NAME, cursorclass=pymysql.cursors.DictCursor
)

# Carica mapping categorie
with open('data/dataset/cat2idx.json') as f:
    cat2idx = json.load(f)
idx2cat = {str(v): k for k, v in cat2idx.items()}

# Carica l'indice pkl per il recupero delle immagini
with open(PKL_PATH, 'rb') as f:
    part_id_map = pickle.load(f)

print("App pronta.")

# --- FUNZIONI HELPER ---

def get_ania_id(cat_idx):
    """ Ottiene il nome della categoria dall'indice. """
    return idx2cat.get(str(cat_idx), "Categoria Sconosciuta")

def get_category_name(ania):
    try:
        with conn.cursor() as cursor:
            sql = f"SELECT descrEx FROM replacement WHERE id = {ania}"
            cursor.execute(sql)
            name = cursor.fetchone()
            return name.get('descrEx') if name else "ND"
    except Exception as e:
        print(f"Errore DB per ania {ania}: {e}")
        return ""

def get_info_from_db(part_id):
    """ Recupera le informazioni di un part_id dal database. """
    try:
        with conn.cursor() as cursor:
            # Esegui una singola query per efficienza se possibile
            sql = f"SELECT dsMar, dsMod, descr FROM component WHERE idmag = {part_id} LIMIT 1"
            cursor.execute(sql)
            part_info = cursor.fetchone()
            return part_info if part_info else {}
    except Exception as e:
        print(f"Errore DB per part_id {part_id}: {e}")
        return {}

def get_images_for_part(target_part_id):
    """ Usa l'indice .pkl per caricare le immagini per un part_id. """
    keys_and_classes = part_id_map.get(str(target_part_id), [])
    if not keys_and_classes:
        return []
    
    keys_to_fetch = {item[0] for item in keys_and_classes}
    
    # Filtra e carica solo le immagini necessarie
    dataset = wds.WebDataset(TRAIN_SHARDS, shardshuffle=False) \
                   .select(lambda sample: sample["__key__"] in keys_to_fetch) \
                   .decode("pil") \
                   .to_tuple("jpg")
    
    return [img[0] for img in dataset]

# --- FUNZIONE PRINCIPALE PER GRADIO ---

def process_search_and_update_ui(input_image):
    """
    Esegue la ricerca e restituisce una lista di aggiornamenti per l'interfaccia Gradio.
    """
    if input_image is None:
        # Se non c'è immagine, restituisce una lista di update per nascondere tutto
        return [gr.update(visible=False) for _ in range(MAX_CATEGORIES * (1 + MAX_PARTS_PER_CAT * 3))]

    # 1. Esegui la ricerca principale per ottenere la lista ordinata di risultati
    params = {
        'top_k_classifier': 5, 'prob_threshold': 0.1, 'faiss_k': 50,
        'alpha': 0.5, 'beta': 1.0, 'gamma': 0.0
    }
    # L'output è una lista di tuple: (part_id, global_score, cat_id, part_score)
    results = search_main(input_image, EMBEDDING_MODEL_PATH, MODEL_PATH, NUM_CLASSES, params)

    # 2. Raggruppa i risultati per categoria, mantenendo l'ordine
    results_by_cat = defaultdict(list)
    ordered_cats = []
    for part_id, global_score, cat_id, part_score in results:
        if cat_id not in results_by_cat:
            ordered_cats.append(cat_id)

        # Recupera info e immagini
        db_info = get_info_from_db(part_id)
        images = get_images_for_part(part_id)
        results_by_cat[cat_id].append({'part_id': part_id, 'global_score': 100 - global_score, 'part_score': 100 - part_score, 'db_info': db_info, 'images': images})

    # 3. Prepara la lista di aggiornamenti per l'interfaccia
    updates = []
    
    # Itera sulle categorie trovate, fino al massimo che possiamo mostrare
    for i in range(MAX_CATEGORIES):
        if i < len(ordered_cats):
            cat_id = ordered_cats[i]
            parts_in_cat = results_by_cat[cat_id]
            
            # Aggiorna la fisarmonica della categoria per renderla visibile e impostare il titolo
            ania = get_ania_id(cat_id)
            cat_name = get_category_name(ania)
            updates.append(gr.update(label=f"{cat_name} (ID: {ania} | Index: {cat_id})", visible=True))

            # Itera sui part_id trovati in questa categoria
            for j in range(MAX_PARTS_PER_CAT):
                if j < len(parts_in_cat):
                    part_data = parts_in_cat[j]
                    part_id = part_data['part_id']
                    score = part_data['part_score']
                    db_info = part_data['db_info']
                    images = part_data['images']
                    
                    # Formatta il testo per il componente Markdown
                    details_text = f"""
                    **Part ID:** {part_id}
                    
                    **Accuratezza:** {score:.2f}%

                    **Veicolo:** {db_info.get('dsMar', 'N/D')} {db_info.get('dsMod', 'N/D')}

                    **Descrizione:** {db_info.get('descr', 'N/D')}
                    """
                    
                    # Aggiungi gli aggiornamenti per il gruppo, il testo e la galleria
                    updates.append(gr.update(visible=True)) # Group
                    updates.append(gr.update(value=details_text)) # Markdown
                    updates.append(gr.update(value=images)) # Gallery
                else:
                    # Nascondi i box dei part_id non usati
                    updates.extend([gr.update(visible=False)] * 3)
        else:
            # Nascondi i box delle categorie non usate
            updates.append(gr.update(visible=False))
            updates.extend([gr.update(visible=False)] * (MAX_PARTS_PER_CAT * 3))
            
    return updates

# --- DEFINIZIONE DELL'INTERFACCIA CON GR.BLOCKS ---

if __name__ == "__main__":
    MAX_CATEGORIES = 3
    MAX_PARTS_PER_CAT = 5

    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚗 Ricerca Visiva Ricambi Auto")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Immagine di Input")
                submit_btn = gr.Button("Cerca Ricambio", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### Risultati della Ricerca")
                
                # Creiamo dinamicamente la struttura di output nidificata
                all_outputs = []
                for i in range(MAX_CATEGORIES):
                    # Ogni categoria ha una sua fisarmonica
                    with gr.Accordion(f"Categoria {i+1}", visible=False) as cat_accordion:
                        all_outputs.append(cat_accordion)
                        # 2. All'interno della fisarmonica, creiamo la griglia a 3 colonne
                        with gr.Row():
                            # Prepariamo le tre colonne
                            grid_columns = [gr.Column(), gr.Column(), gr.Column()]
                        
                        # Lista per i componenti interni di questa categoria
                        part_outputs_in_cat = []
                        for j in range(MAX_PARTS_PER_CAT):
                            # Determina in quale colonna inserire il box del part_id
                            target_column = grid_columns[j % 3]
                            
                            with target_column:
                                # 3. Ogni part_id ha il suo box con dettagli e galleria
                                with gr.Group(visible=False) as part_group:
                                    details_md = gr.Markdown()
                                    image_gallery = gr.Gallery(
                                        label="Immagini", columns=4, height="auto",
                                        object_fit="contain", preview=True
                                    )
                                    # Aggiungiamo i componenti alla lista flat generale
                                    all_outputs.extend([part_group, details_md, image_gallery])
        
        # Collega il bottone alla funzione
        submit_btn.click(
            fn=process_search_and_update_ui,
            inputs=image_input,
            outputs=all_outputs
        )
    
    app.launch(share=True)