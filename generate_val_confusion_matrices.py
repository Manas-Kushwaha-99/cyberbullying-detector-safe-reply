"""Generate validation set confusion matrices for all models."""
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import DEVICE, MODELS_DIR, FIGURES_DIR, LABELS
from src.data_split import split_dataset
from src.preprocessing import build_preprocessed_dataset
from src.evaluation.core_metrics import evaluate_classifier
from sklearn.metrics import confusion_matrix

os.makedirs(FIGURES_DIR, exist_ok=True)

# Load validation data
train_df, val_df, test_df = split_dataset()
val_df_proc = build_preprocessed_dataset(val_df.copy(), for_transformer=False)
val_df_tx = build_preprocessed_dataset(val_df.copy(), for_transformer=True)
val_texts = val_df_proc["processed_text"].tolist()
val_texts_tx = val_df_tx["processed_text"].tolist()
val_labels = val_df_proc["label"].tolist()

models_to_eval = [
    ("TF-IDF+SVM", None, val_texts, val_labels),
    ("LSTM", None, val_texts, val_labels),
    ("DistilBERT", None, val_texts_tx, val_df_tx["label"].tolist()),
    ("DistilBERT+LoRA", None, val_texts_tx, val_df_tx["label"].tolist()),
]

# Load models
from src.models.tfidf_svm import TfidfSvmClassifier
from src.models.lstm_model import CyberbullyingLSTM, LSTMWrapper
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.models.transformer_wrappers import DistilBERTWrapper
from peft import PeftModel
from src.models.transformer_wrappers import DistilBERTLoRAWrapper

# TF-IDF+SVM
clf = TfidfSvmClassifier.load(os.path.join(MODELS_DIR, "tfidf_svm.pkl"))
models_to_eval[0] = ("TF-IDF+SVM", clf, val_texts, val_labels)

# LSTM
checkpoint = torch.load(os.path.join(MODELS_DIR, "lstm.pt"), map_location=DEVICE)
model = CyberbullyingLSTM(**checkpoint["config"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
with open(os.path.join(MODELS_DIR, "lstm_vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)
wrapped_lstm = LSTMWrapper(model, vocab, LABELS)
models_to_eval[1] = ("LSTM", wrapped_lstm, val_texts, val_labels)

# DistilBERT
db_tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join(MODELS_DIR, "distilbert"))
db_model = DistilBertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, "distilbert")).to(DEVICE)
wrapped_db = DistilBERTWrapper(db_model, db_tokenizer, {i: l for i, l in enumerate(LABELS)})
models_to_eval[2] = ("DistilBERT", wrapped_db, val_texts_tx, val_df_tx["label"].tolist())

# DistilBERT+LoRA
dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join(MODELS_DIR, "distilbert_lora"))
base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABELS))
dbl_model = PeftModel.from_pretrained(base_model, os.path.join(MODELS_DIR, "distilbert_lora")).to(DEVICE)
wrapped_dbl = DistilBERTLoRAWrapper(dbl_model, dbl_tokenizer, {i: l for i, l in enumerate(LABELS)})
models_to_eval[3] = ("DistilBERT+LoRA", wrapped_dbl, val_texts_tx, val_df_tx["label"].tolist())

# Generate confusion matrices
for name, model, texts, labels in models_to_eval:
    print(f"[Generating] Val confusion matrix for {name}...")
    
    # Get predictions
    if hasattr(model, "predict_labels"):
        preds = model.predict_labels(texts)
    else:
        preds = model.predict(texts)
        if isinstance(preds[0], (int, np.integer)):
            preds = [LABELS[p] for p in preds]
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds, labels=LABELS)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_title(f"Confusion Matrix: {name} (Validation Set)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    
    # Save
    safe_name = name.lower().replace("+", "_").replace("-", "_")
    fig_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}_val.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {fig_path}")

print("\nAll validation confusion matrices generated.")
