import pandas as pd
from tqdm import tqdm
import itertools
import torch
from PIL import Image

# Importa le funzioni modificate da search.py
from search import load_models, extract_embedding, predict_category, query_faiss, transform
from config import * # Importa le tue configurazioni (DEVICE, MODELS_PATH, etc.)

def run_search_with_params(img_tensor, emb_model, cls_model, params):
    """
    Esegue la logica di ricerca completa usando un set di parametri.
    Questa funzione è un adattamento della logica 'main' di search.py.
    """
    embedding = extract_embedding(img_tensor, emb_model)
    categories = predict_category(img_tensor, cls_model, k=params['top_k_classifier'])
    
    all_results = []
    for cat_id, prob in categories:
        if prob > params['prob_threshold']:
            results = query_faiss(embedding, cat_id, 
                                  faiss_k=params['faiss_k'], 
                                  alpha=params['alpha'], 
                                  beta=params['beta'], 
                                  gamma=params['gamma'])
            for part_id, distance in results:
                all_results.append(part_id) # Ci interessa solo il part_id predetto
                
    # Restituisce una lista flat di part_id predetti dai migliori candidati
    return all_results

if __name__ == "__main__":
    # 1. DEFINISCI LA GRIGLIA DI IPERPARAMETRI
    param_grid = {
        'top_k_classifier': [1, 3],
        'prob_threshold': [0.05, 0.1],
        'faiss_k': [30, 50],
        'alpha': [1.0, 0.5], # Peso per la distanza media
        'beta': [1.0, 0.5],  # Peso per la distanza minima
        'gamma': [0.1, 0.0]  # Peso per la varianza della distanza
    }

    # Carica i modelli una sola volta
    print("Caricamento modelli...")
    emb_model, cls_model = load_models(EMBEDDING_MODEL_PATH, MODEL_PATH, NUM_CLASSES)

    # Carica il tuo CSV di test
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    # Crea tutte le combinazioni di parametri
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = -1
    best_params = None

    print(f"Inizio Grid Search su {len(param_combinations)} combinazioni...")

    # 2. PROVA OGNI COMBINAZIONE
    for params in tqdm(param_combinations, desc="Grid Search"):
        correct_predictions = 0
        
        # 3. VALUTA SUL DATASET DI TEST
        for index, row in test_df.iterrows():
            image_path = row['image_path']
            true_part_id = str(row['part_id'])
            
            try:
                img = Image.open(image_path).convert("RGB")
                img_tensor = transform(img)
                
                # Esegui la ricerca con i parametri correnti
                predicted_part_ids = run_search_with_params(img_tensor, emb_model, cls_model, params)
                
                # Calcola se la predizione è corretta (Recall@N)
                # In questo caso, controlliamo se il part_id corretto è tra i risultati
                if true_part_id in predicted_part_ids:
                    correct_predictions += 1
            except Exception as e:
                print(f"Errore processando {image_path}: {e}")

        # Calcola lo score per questa combinazione (accuratezza)
        score = correct_predictions / len(test_df)
        
        # 4. SALVA LA COMBINAZIONE MIGLIORE
        if score > best_score:
            best_score = score
            best_params = params
            print(f"\n🚀 Nuovo score migliore: {best_score:.4f} con parametri: {best_params}")

    print("\n--- Grid Search Completata ---")
    print(f"Miglior score (Recall): {best_score:.4f}")
    print(f"Migliori parametri: {best_params}")