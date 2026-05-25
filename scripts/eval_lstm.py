"""Evaluate saved LSTM model."""
import os
import sys
import json
import pickle

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

from src.data_split import split_dataset
from src.preprocessing import build_preprocessed_dataset
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    SEED, DEVICE, MODELS_DIR, REPORTS_DIR,
    LSTM_MAX_LEN, LSTM_BATCH_SIZE, LABELS, NUM_CLASSES, set_seed
)
from src.models.lstm_model import CyberbullyingLSTM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

set_seed(SEED)

# Load data
train_df, val_df, test_df = split_dataset(csv_path='data/synthetic/cb_enhanced_v2.csv')
test_df = build_preprocessed_dataset(test_df, for_transformer=False)
X_test = test_df["processed_text"].tolist()
y_test = test_df["label"].tolist()

# Load vocab
vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

label2idx = {lbl: i for i, lbl in enumerate(LABELS)}

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

test_ds = TextDataset(X_test, y_test, vocab, LSTM_MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=LSTM_BATCH_SIZE)

# Load model
from src.config import LSTM_EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_BIDIRECTIONAL
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
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), weights_only=True))
model.eval()

test_preds = []
test_labels = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(batch_y.numpy())

test_pred_labels = [LABELS[p] for p in test_preds]
test_true_labels = [LABELS[y] for y in test_labels]

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
}

os.makedirs(REPORTS_DIR, exist_ok=True)
report_path = os.path.join(REPORTS_DIR, "lstm_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"[Saved] Report -> {report_path}")

# Load history and merge
history_path = os.path.join(REPORTS_DIR, "lstm_history.json")
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)
    results["history"] = history
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nLSTM Evaluation:")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1: {f1:.4f}")
