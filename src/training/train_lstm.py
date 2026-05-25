"""Training script for LSTM."""
import os
import time
import json
import pickle
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import (
    SEED, DEVICE, MODELS_DIR, CHECKPOINTS_DIR, REPORTS_DIR,
    LSTM_EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_BIDIRECTIONAL, LSTM_MAX_LEN, LSTM_BATCH_SIZE, LSTM_LR,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_MIN_DELTA, LSTM_GLOVE_PATH,
    LABELS, NUM_CLASSES
)
from src.preprocessing import build_preprocessed_dataset
from src.data_split import split_dataset
from src.models.lstm_model import CyberbullyingLSTM, LSTMWrapper
from src.evaluation.core_metrics import evaluate_classifier
from sklearn.metrics import f1_score


class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        self.label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize by whitespace
        tokens = text.split()
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        
        # Pad / truncate
        if len(indices) < self.max_len:
            indices += [self.word2idx["<PAD>"]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices, dtype=torch.long), \
               torch.tensor(self.label2idx[label], dtype=torch.long)


def build_vocab(texts, min_freq=2):
    """Build vocabulary from training texts."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


def load_glove_embeddings(glove_path, vocab, embed_dim):
    """Load GloVe embeddings and build embedding matrix."""
    embeddings_index = {}
    if not os.path.exists(glove_path):
        print(f"[Warning] GloVe file not found at {glove_path}. Using random embeddings.")
        return None
    
    print(f"[Loading] GloVe embeddings from {glove_path}...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    
    embedding_matrix = np.random.randn(len(vocab), embed_dim).astype(np.float32) * 0.01
    embedding_matrix[vocab["<PAD>"]] = np.zeros(embed_dim, dtype=np.float32)
    
    found = 0
    for word, idx in vocab.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
            found += 1
    
    print(f"[GloVe] Found {found}/{len(vocab)} words in embeddings.")
    return torch.tensor(embedding_matrix, dtype=torch.float32)


def train_lstm(csv_path=None, save_model=True):
    from src.config import set_seed
    set_seed(SEED)
    
    print("=" * 60)
    print("Training LSTM")
    print("=" * 60)
    
    # Load & split data
    train_df, val_df, test_df = split_dataset(csv_path=csv_path)
    
    # Preprocess
    train_df = build_preprocessed_dataset(train_df, for_transformer=False)
    val_df = build_preprocessed_dataset(val_df, for_transformer=False)
    test_df = build_preprocessed_dataset(test_df, for_transformer=False)
    
    X_train = train_df["processed_text"].tolist()
    y_train = train_df["label"].tolist()
    X_val = val_df["processed_text"].tolist()
    y_val = val_df["label"].tolist()
    X_test = test_df["processed_text"].tolist()
    y_test = test_df["label"].tolist()
    
    # Build vocab
    vocab = build_vocab(X_train, min_freq=2)
    vocab_size = len(vocab)
    print(f"[LSTM] Vocab size: {vocab_size}")
    
    # Load embeddings
    pretrained_emb = load_glove_embeddings(LSTM_GLOVE_PATH, vocab, LSTM_EMBED_DIM)
    
    # Datasets & Loaders
    train_dataset = TextDataset(X_train, y_train, vocab, LSTM_MAX_LEN)
    val_dataset = TextDataset(X_val, y_val, vocab, LSTM_MAX_LEN)
    test_dataset = TextDataset(X_test, y_test, vocab, LSTM_MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=LSTM_BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE,
                             shuffle=False, num_workers=0)
    
    # Model
    model = CyberbullyingLSTM(
        vocab_size=vocab_size,
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=LSTM_NUM_LAYERS,
        bidirectional=LSTM_BIDIRECTIONAL,
        dropout=LSTM_DROPOUT,
        pretrained_embeddings=pretrained_emb,
        freeze_embeddings=False
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    
    # Training loop with early stopping
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    best_val_f1 = -1
    best_state = None
    patience_counter = 0
    
    total_start = time.time()
    
    for epoch in range(1, LSTM_EPOCHS + 1):
        # Training
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        val_preds_all = []
        val_true_all = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_losses.append(loss.item())
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
                val_preds_all.extend(preds.cpu().tolist())
                val_true_all.extend(batch_y.cpu().tolist())
        
        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_true_all, val_preds_all, average="macro", zero_division=0)
        
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model based on validation F1
        if val_f1 > best_val_f1 + LSTM_MIN_DELTA:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  -> New best val F1: {val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= LSTM_PATIENCE:
            print(f"[Early stopping] No improvement for {LSTM_PATIENCE} epochs.")
            break
    
    train_time = time.time() - total_start
    print(f"[LSTM] Training completed in {train_time:.2f}s")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Save model
    if save_model:
        model_path = os.path.join(MODELS_DIR, "lstm.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {
                "vocab_size": vocab_size,
                "embed_dim": LSTM_EMBED_DIM,
                "hidden_dim": LSTM_HIDDEN_DIM,
                "num_classes": NUM_CLASSES,
                "num_layers": LSTM_NUM_LAYERS,
                "bidirectional": LSTM_BIDIRECTIONAL,
                "dropout": LSTM_DROPOUT,
            }
        }, model_path)
        print(f"[Saved] Model -> {model_path}")
        
        # Save vocab separately
        vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
    
    # Save history
    history_path = os.path.join(REPORTS_DIR, "lstm_history.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Saved] History -> {history_path}")
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_probs.extend(probs.cpu().tolist())
    
    # Wrap for evaluator
    wrapped = LSTMWrapper(model, vocab, LABELS)
    results = evaluate_classifier(wrapped, X_test, y_test, model_name="LSTM")
    results["train_time_sec"] = float(train_time)
    results["history"] = history
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, "lstm_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Report -> {report_path}")
    
    return model, results


if __name__ == "__main__":
    train_lstm()
