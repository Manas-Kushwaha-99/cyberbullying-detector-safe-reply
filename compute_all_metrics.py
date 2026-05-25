"""Compute consolidated Train/Val/Test metrics for all models."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_split import split_dataset
from src.preprocessing import build_preprocessed_dataset
from src.config import DEVICE, LABELS, BERT_MAX_LEN, MODELS_DIR, REPORTS_DIR

# Load data splits
train_df, val_df, test_df = split_dataset()

# Prepare datasets
train_df_proc = build_preprocessed_dataset(train_df.copy(), for_transformer=False)
val_df_proc = build_preprocessed_dataset(val_df.copy(), for_transformer=False)
test_df_proc = build_preprocessed_dataset(test_df.copy(), for_transformer=False)

train_texts = train_df_proc["processed_text"].tolist()
val_texts = val_df_proc["processed_text"].tolist()
test_texts = test_df_proc["processed_text"].tolist()

results = {}

# ═══════════════════════════════════════════════════════════════════
# 1. TF-IDF + SVM
# ═══════════════════════════════════════════════════════════════════
print("Evaluating TF-IDF + SVM on Train/Val/Test...")
from src.models.tfidf_svm import TfidfSvmClassifier
from src.evaluation.core_metrics import evaluate_classifier

clf = TfidfSvmClassifier.load(os.path.join(MODELS_DIR, "tfidf_svm.pkl"))

train_res = evaluate_classifier(clf, train_texts, train_df_proc["label"].tolist(), model_name="TF-IDF+SVM-Train")
val_res = evaluate_classifier(clf, val_texts, val_df_proc["label"].tolist(), model_name="TF-IDF+SVM-Val")
test_res = evaluate_classifier(clf, test_texts, test_df_proc["label"].tolist(), model_name="TF-IDF+SVM-Test")

results["TF-IDF+SVM"] = {
    "train_acc": train_res["accuracy"],
    "train_f1": train_res["f1_macro"],
    "val_acc": val_res["accuracy"],
    "val_f1": val_res["f1_macro"],
    "test_acc": test_res["accuracy"],
    "test_f1": test_res["f1_macro"],
    "test_precision": test_res["precision_macro"],
    "test_recall": test_res["recall_macro"],
}
print(f"  Train Acc: {train_res['accuracy']:.4f}, Val Acc: {val_res['accuracy']:.4f}, Test Acc: {test_res['accuracy']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 2. LSTM
# ═══════════════════════════════════════════════════════════════════
print("Evaluating LSTM on Train/Val/Test...")
import torch, pickle
from src.models.lstm_model import CyberbullyingLSTM, LSTMWrapper

checkpoint = torch.load(os.path.join(MODELS_DIR, "lstm.pt"), map_location=DEVICE)
model = CyberbullyingLSTM(**checkpoint["config"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
with open(os.path.join(MODELS_DIR, "lstm_vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

wrapped = LSTMWrapper(model, vocab, LABELS)
train_res = evaluate_classifier(wrapped, train_texts, train_df_proc["label"].tolist(), model_name="LSTM-Train")
val_res = evaluate_classifier(wrapped, val_texts, val_df_proc["label"].tolist(), model_name="LSTM-Val")
test_res = evaluate_classifier(wrapped, test_texts, test_df_proc["label"].tolist(), model_name="LSTM-Test")

results["LSTM"] = {
    "train_acc": train_res["accuracy"],
    "train_f1": train_res["f1_macro"],
    "val_acc": val_res["accuracy"],
    "val_f1": val_res["f1_macro"],
    "test_acc": test_res["accuracy"],
    "test_f1": test_res["f1_macro"],
    "test_precision": test_res["precision_macro"],
    "test_recall": test_res["recall_macro"],
}
print(f"  Train Acc: {train_res['accuracy']:.4f}, Val Acc: {val_res['accuracy']:.4f}, Test Acc: {test_res['accuracy']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 3. DistilBERT
# ═══════════════════════════════════════════════════════════════════
print("Evaluating DistilBERT on Train/Val/Test...")
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.models.transformer_wrappers import DistilBERTWrapper

train_df_tx = build_preprocessed_dataset(train_df.copy(), for_transformer=True)
val_df_tx = build_preprocessed_dataset(val_df.copy(), for_transformer=True)
test_df_tx = build_preprocessed_dataset(test_df.copy(), for_transformer=True)

train_texts_tx = train_df_tx["processed_text"].tolist()
val_texts_tx = val_df_tx["processed_text"].tolist()
test_texts_tx = test_df_tx["processed_text"].tolist()

db_tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join(MODELS_DIR, "distilbert"))
db_model = DistilBertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, "distilbert")).to(DEVICE)

wrapped = DistilBERTWrapper(db_model, db_tokenizer, {i: l for i, l in enumerate(LABELS)})
train_res = evaluate_classifier(wrapped, train_texts_tx, train_df_tx["label"].tolist(), model_name="DistilBERT-Train")
val_res = evaluate_classifier(wrapped, val_texts_tx, val_df_tx["label"].tolist(), model_name="DistilBERT-Val")
test_res = evaluate_classifier(wrapped, test_texts_tx, test_df_tx["label"].tolist(), model_name="DistilBERT-Test")

results["DistilBERT"] = {
    "train_acc": train_res["accuracy"],
    "train_f1": train_res["f1_macro"],
    "val_acc": val_res["accuracy"],
    "val_f1": val_res["f1_macro"],
    "test_acc": test_res["accuracy"],
    "test_f1": test_res["f1_macro"],
    "test_precision": test_res["precision_macro"],
    "test_recall": test_res["recall_macro"],
}
print(f"  Train Acc: {train_res['accuracy']:.4f}, Val Acc: {val_res['accuracy']:.4f}, Test Acc: {test_res['accuracy']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 4. DistilBERT + LoRA
# ═══════════════════════════════════════════════════════════════════
print("Evaluating DistilBERT + LoRA on Train/Val/Test...")
from peft import PeftModel
from src.models.transformer_wrappers import DistilBERTLoRAWrapper

dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join(MODELS_DIR, "distilbert_lora"))
base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABELS))
dbl_model = PeftModel.from_pretrained(base_model, os.path.join(MODELS_DIR, "distilbert_lora")).to(DEVICE)

wrapped = DistilBERTLoRAWrapper(dbl_model, dbl_tokenizer, {i: l for i, l in enumerate(LABELS)})
train_res = evaluate_classifier(wrapped, train_texts_tx, train_df_tx["label"].tolist(), model_name="DistilBERT+LoRA-Train")
val_res = evaluate_classifier(wrapped, val_texts_tx, val_df_tx["label"].tolist(), model_name="DistilBERT+LoRA-Val")
test_res = evaluate_classifier(wrapped, test_texts_tx, test_df_tx["label"].tolist(), model_name="DistilBERT+LoRA-Test")

results["DistilBERT+LoRA"] = {
    "train_acc": train_res["accuracy"],
    "train_f1": train_res["f1_macro"],
    "val_acc": val_res["accuracy"],
    "val_f1": val_res["f1_macro"],
    "test_acc": test_res["accuracy"],
    "test_f1": test_res["f1_macro"],
    "test_precision": test_res["precision_macro"],
    "test_recall": test_res["recall_macro"],
}
print(f"  Train Acc: {train_res['accuracy']:.4f}, Val Acc: {val_res['accuracy']:.4f}, Test Acc: {test_res['accuracy']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# Save consolidated results
# ═══════════════════════════════════════════════════════════════════
consolidated_path = os.path.join(REPORTS_DIR, "consolidated_results.json")
with open(consolidated_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nConsolidated results saved: {consolidated_path}")

# Generate Markdown table
table_lines = [
    "# Consolidated Results: Train / Validation / Test Metrics",
    "",
    "| Model | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test Precision | Test Recall | Test F1 |",
    "|-------|-----------|----------|---------|--------|----------|----------------|-------------|---------|",
]

for model_name, r in results.items():
    table_lines.append(
        f"| {model_name} | {r['train_acc']:.4f} | {r['train_f1']:.4f} | "
        f"{r['val_acc']:.4f} | {r['val_f1']:.4f} | {r['test_acc']:.4f} | "
        f"{r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_f1']:.4f} |"
    )

table_path = os.path.join("tables", "consolidated_results.md")
with open(table_path, "w", encoding="utf-8") as f:
    f.write("\n".join(table_lines))
print(f"Markdown table saved: {table_path}")

# Print to console
print("\n" + "="*80)
print("CONSOLIDATED RESULTS")
print("="*80)
print(f"{'Model':<20} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Test F1':>10}")
print("-"*80)
for model_name, r in results.items():
    print(f"{model_name:<20} {r['train_acc']:>10.4f} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f} {r['test_f1']:>10.4f}")
print("="*80)
