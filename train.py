import torch
from tqdm import tqdm
from config import *
from dataset.loader import get_dataloader, get_val_dataloader
from model.classifier import get_model
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch.optim as optim
import os
import optuna
import argparse

# --- NUOVE IMPOSTAZIONI PER IL FINE-TUNING ---
# Numero di epoche per cui addestrare solo il classificatore (con il resto del modello congelato)
FROZEN_EPOCHS = 5 

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

# --- La logica di training è ora dentro la funzione "objective" per Optuna ---
# In train.py

def objective(trial):
    """
    Versione aggiornata che mostra il progresso delle epoche per ogni trial.
    """
    # 1. Definisci gli iperparametri (nessun cambiamento qui)
    lr_frozen = trial.suggest_float("lr_frozen", 1e-4, 1e-2, log=True)
    unfrozen_lr_factor = trial.suggest_categorical("unfrozen_lr_factor", [5.0, 10.0, 20.0, 50.0])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

    # Carica dataloader e modello
    train_loader = get_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS)
    val_loader = get_val_dataloader(VAL_SHARDS, BATCH_SIZE, NUM_WORKERS)
    model = get_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0

    # -----------------------------------------------------
    # FASE 1: Addestramento del solo classificatore
    # -----------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FROZEN_EPOCHS)

    # Aggiungiamo una barra di avanzamento per le epoche della FASE 1
    pbar_frozen = tqdm(range(FROZEN_EPOCHS), desc=f"Trial {trial.number} Fase 1", leave=False)
    for epoch in pbar_frozen:
        model.train()
        for images, labels in train_loader:
            # ... logica di training ...
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # -----------------------------------------------------
    # FASE 2: Fine-tuning di tutto il modello
    # -----------------------------------------------------
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=lr_frozen / unfrozen_lr_factor, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Aggiungiamo una barra di avanzamento per le epoche della FASE 2
    pbar_unfrozen = tqdm(range(NUM_EPOCHS), desc=f"Trial {trial.number} Fase 2", leave=False)
    for epoch in pbar_unfrozen:
        model.train()
        for images, labels in train_loader:
            # ... logica di training ...
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Aggiorna la barra di avanzamento con l'accuratezza corrente
        pbar_unfrozen.set_postfix(val_acc=f"{val_acc:.4f}")
            
        trial.report(val_acc, epoch)
        if trial.should_prune():
            # Chiudi la pbar prima di sollevare l'eccezione
            pbar_unfrozen.close()
            raise optuna.exceptions.TrialPruned()

    return best_val_acc

def study_model():
    # --- MODIFICHE PER LA BARRA DI AVANZAMENTO ---
    N_TRIALS = 20  # Definisci il numero di prove qui

    # La funzione callback che aggiornerà la barra dopo ogni trial
    def callback(study, trial):
        pbar.update(1)

    # Inizializza la barra di avanzamento di tqdm
    pbar = tqdm(total=N_TRIALS, desc="Ottimizzazione Parametri")
    # ----------------------------------------------

    # Crea e lancia lo "studio" di Optuna
    study = optuna.create_study(direction="maximize")
    
    try:
        # Aggiungi il 'callback' alla chiamata optimize
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])
    except KeyboardInterrupt:
        print("Ottimizzazione interrotta manualmente.")
    finally:
        # Assicurati di chiudere la barra di avanzamento alla fine
        pbar.close()
    
    # Stampa i risultati finali
    print("\n--- Ottimizzazione Completata ---")
    print(f"Miglior score (accuratezza): {study.best_value:.4f}")
    print("Migliori iperparametri:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

def train_model():
    # Carica i dataloader
    train_loader = get_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS)
    val_loader = get_val_dataloader(VAL_SHARDS, BATCH_SIZE, NUM_WORKERS)
    
    # Carica il modello
    model = get_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0

    # -----------------------------------------------------
    # FASE 1: Addestramento del solo classificatore
    # -----------------------------------------------------
    print("--- FASE 1: Addestramento del classificatore (backbone congelato) ---")
    
    # Congela tutti i layer del modello
    for param in model.parameters():
        param.requires_grad = False
    
    # Scongela solo i layer del nostro classificatore custom 'fc'
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # L'optimizer riceve solo i parametri che abbiamo scongelato
    # optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CLASS_LEARNING_RATE, weight_decay=CLASS_WEIGHT_DECAY) 
    # Lo scheduler monitorerà l'accuratezza per regolare il learning rate
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='max')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FROZEN_EPOCHS) 

    for epoch in range(FROZEN_EPOCHS):
        model.train()
        running_loss, num_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f"FASE 1 - Epoch {epoch+1}/{FROZEN_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=running_loss / num_batches)

        val_acc = evaluate(model, val_loader)
        tqdm.write(f"FASE 1 - Epoch {epoch+1} Loss: {running_loss/num_batches:.4f} | Acc: {val_acc:.4f}")
        # scheduler.step(val_acc)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            tqdm.write(f"🚀 Modello salvato in {MODEL_PATH} con vall acc: {best_val_acc:.4f}")

    # -----------------------------------------------------
    # FASE 2: Fine-tuning di tutto il modello
    # -----------------------------------------------------
    print("\n--- FASE 2: Fine-tuning dell'intero modello (backbone scongelato) ---")
    
    # Scongela tutti i layer
    for param in model.parameters():
        param.requires_grad = True
    
    # Crea un nuovo optimizer con un learning rate più basso per tutti i parametri
    # optimizer = Adam(model.parameters(), lr=LEARNING_RATE / UNFROZEN_LR_FACTOR)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='max')
    optimizer = AdamW(model.parameters(), lr=CLASS_LEARNING_RATE / CLASS_UNFROZEN_LR_FACTOR, weight_decay=CLASS_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) 

    # Continua il training per le restanti epoche
    for epoch in range(NUM_EPOCHS): # NUM_EPOCHS ora definisce le epoche solo per questa fase
        model.train()
        running_loss, num_batches = 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"FASE 2 - Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=running_loss / num_batches)

        val_acc = evaluate(model, val_loader)
        tqdm.write(f"FASE 2 - Epoch {epoch+1} Loss: {running_loss/num_batches:.4f} | Acc: {val_acc:.4f}")
        # scheduler.step(val_acc)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            tqdm.write(f"🚀 Modello salvato in {MODEL_PATH} con vall acc: {best_val_acc:.4f}")

    print("\nTraining completato!")
    print(f"Miglior Validation Accuracy ottenuta: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="'study' or 'train''")
    
    args = parser.parse_args()

    if args.action == 'study':
        study_model()
    elif args.action == 'train':
        train_model()