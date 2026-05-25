"""Retrain LSTM on enhanced dataset with correct import order."""
import os
import sys
import json
import time
import pickle
from collections import Counter

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

# CRITICAL: Import sklearn/pandas BEFORE torch
from src.data_split import split_dataset
from src.preprocessing import build_preprocessed_dataset

# Now safe to import torch and config
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import (
    SEED, DEVICE, MODELS_DIR, CHECKPOINTS_DIR, REPORTS_DIR,
    LSTM_EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_BIDIRECTIONAL, LSTM_MAX_LEN, LSTM_BATCH_SIZE, LSTM_LR,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_MIN_DELTA, LSTM_GLOVE_PATH,
    LABELS, NUM_CLASSES, set_seed
)
from src.evaluation.core_metrics import evaluate_classifier
from sklearn.metrics import f1_score

set_seed(SEED)

print("=" * 60)
print("Training LSTM (Enhanced Dataset)")
print("=" * 60)

# Load & split data
train_df, val_df, test_df = split_dataset(csv_path='data/synthetic/cb_enhanced_v2.csv')

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
print("[LSTM] Building vocabulary...")
counter = Counter()
for text in X_train:
    counter.update(text.split())

vocab = {"<PAD>": 0, "<UNK>": 1}
idx = 2
min_freq = 2
for word, freq in counter.items():
    if freq >= min_freq:
        vocab[word] = idx
        idx += 1

print(f"[LSTM] Vocab size: {len(vocab)}")

# Save vocab
vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
os.makedirs(MODELS_DIR, exist_ok=True)
with open(vocab_path, "wb") as f:
    pickle.dump(vocab, f)
print(f"[Saved] Vocab -> {vocab_path}")

# Label mapping
label2idx = {lbl: i for i, lbl in enumerate(LABELS)}

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        self.label2idx = label2idx
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = text.split()
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        if len(indices) < self.max_len:
            indices += [self.word2idx["<PAD>"]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.label2idx[label], dtype=torch.long)

train_ds = TextDataset(X_train, y_train, vocab, LSTM_MAX_LEN)
val_ds = TextDataset(X_val, y_val, vocab, LSTM_MAX_LEN)
test_ds = TextDataset(X_test, y_test, vocab, LSTM_MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=LSTM_BATCH_SIZE)

# Model
from src.models.lstm_model import CyberbullyingLSTM

model = CyberbullyingLSTM(
    vocab_size=len(vocab),
    embed_dim=LSTM_EMBED_DIM,
    hidden_dim=LSTM_HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    num_layers=LSTM_NUM_LAYERS,
    dropout=LSTM_DROPOUT,
    bidirectional=LSTM_BIDIRECTIONAL,
    pretrained_embeddings=None
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)

# Training loop
best_val_f1 = 0
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_f1": [], "train_acc": []}

print(f"[LSTM] Training on {len(X_train)} samples for up to {LSTM_EPOCHS} epochs...")
start_time = time.time()

for epoch in range(LSTM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    train_acc = correct / total
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(batch_y.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_f1"].append(val_f1)
    history["train_acc"].append(train_acc)
    
    print(f"[Epoch {epoch+1}/{LSTM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    # Early stopping
    if val_f1 > best_val_f1 + LSTM_MIN_DELTA:
        best_val_f1 = val_f1
        patience_counter = 0
        # Save best model
        model_path = os.path.join(MODELS_DIR, "lstm.pt")
        torch.save(model.state_dict(), model_path)
        print(f"  -> New best Val F1: {best_val_f1:.4f} (saved)")
    else:
        patience_counter += 1
        if patience_counter >= LSTM_PATIENCE:
            print(f"  -> Early stopping at epoch {epoch+1}")
            break

train_time = time.time() - start_time
print(f"[LSTM] Training completed in {train_time:.1f}s")

# Load best model for evaluation
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), map_location='cpu'))
model.eval()

# Test evaluation
test_preds = []
test_labels = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(batch_y.numpy())

# Convert to string labels
test_pred_labels = [LABELS[p] for p in test_preds]
test_true_labels = [LABELS[y] for y in test_labels]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc = accuracy_score(test_true_labels, test_pred_labels)
prec = precision_score(test_true_labels, test_pred_labels, average="macro", zero_division=0)
rec = recall_score(test_true_labels, test_pred_labels, average="macro", zero_division=0)
f1 = f1_score(test_true_labels, test_pred_labels, average="macro", zero_division=0)
f1w = f1_score(test_true_labels, test_pred_labels, average="weighted", zero_division=0)
cm = confusion_matrix(test_true_labels, test_pred_labels, labels=LABELS)
report = classification_report(test_true_labels, test_pred_labels, labels=LABELS, output_dict=True, zero_division=0)
report_str = classification_report(test_true_labels, test_pred_labels, labels=LABELS, zero_division=0)

results = {
    "model": "LSTM",
    "accuracy": float(acc),
    "precision_macro": float(prec),
    "recall_macro": float(rec),
    "f1_macro": float(f1),
    "f1_weighted": float(f1w),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "classification_report_str": report_str,
    "train_time_sec": train_time,
    "history": history,
    "best_val_f1": float(best_val_f1)
}

os.makedirs(REPORTS_DIR, exist_ok=True)
report_path = os.path.join(REPORTS_DIR, "lstm_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"[Saved] Report -> {report_path}")

# Save history separately
history_path = os.path.join(REPORTS_DIR, "lstm_history.json")
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)
print(f"[Saved] History -> {history_path}")

print("\nLSTM retraining complete!")
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1: {results['f1_macro']:.4f}")
