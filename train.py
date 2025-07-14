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
# Fattore di riduzione del learning rate per la seconda fase di fine-tuning
UNFROZEN_LR_FACTOR = 10.0

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

# --- NUOVA FUNZIONE "OBJECTIVE" PER OPTUNA ---
def objective(trial):
    """
    Questa funzione esegue un ciclo di addestramento completo e restituisce 
    lo score che Optuna cercherà di massimizzare.
    """
    # 1. Definisci lo spazio di ricerca degli iperparametri
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    
    # Carica dataloader e modello
    train_loader = get_dataloader(TRAIN_SHARDS, BATCH_SIZE, NUM_WORKERS)
    val_loader = get_val_dataloader(VAL_SHARDS, BATCH_SIZE, NUM_WORKERS)
    model = get_model(NUM_CLASSES).to(DEVICE)

    # Usa i parametri suggeriti da Optuna
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_acc = 0.0

    print(f"\n--- Inizio Trial #{trial.number} con lr={learning_rate:.6f}, wd={weight_decay:.6f} ---")
    
    # Ciclo di training (puoi anche ridurre NUM_EPOCHS qui per fare trial più veloci)
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, num_batches = 0.0, 0
        
        # Usiamo 'disable=True' in tqdm per non inondare il log durante i trial
        pbar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}", disable=True) 
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        
        scheduler.step()

        # Valutazione
        val_acc = evaluate(model, val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        # Optuna può interrompere i trial che non sembrano promettenti (pruning)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # 2. Restituisci il valore che Optuna deve massimizzare
    return best_val_acc

def study_model():
    # 3. Crea e lancia lo "studio" di Optuna
    # La direction è "maximize" perché vogliamo massimizzare l'accuratezza
    study = optuna.create_study(direction="maximize")
    
    # Avvia l'ottimizzazione. Optuna chiamerà la funzione 'objective' 20 volte.
    study.optimize(objective, n_trials=20) 
    
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
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE) 
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
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE / UNFROZEN_LR_FACTOR)
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