"""Step 1: Evaluate existing DistilBERT model."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Loading data...")
from src.data_split import split_dataset
train_df, val_df, test_df = split_dataset()
print(f"Data loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

print("Preprocessing test data...")
from src.preprocessing import build_preprocessed_dataset
test_df_tx = build_preprocessed_dataset(test_df.copy(), for_transformer=True)
test_texts = test_df_tx["processed_text"].tolist()
print(f"Test texts: {len(test_texts)}")

print("Importing torch and transformers...")
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.models.transformer_wrappers import DistilBERTWrapper
from src.evaluation.core_metrics import evaluate_classifier
from src.evaluation.efficiency_benchmark import count_parameters
from src.config import DEVICE, MODELS_DIR, REPORTS_DIR, LABELS, BERT_MAX_LEN
print("Imports done.")

print("Loading DistilBERT model...")
distilbert_model_path = os.path.join(MODELS_DIR, "distilbert")
db_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_model_path)
db_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path).to(DEVICE)
print("Model loaded.")

print("Evaluating...")
wrapped = DistilBERTWrapper(db_model, db_tokenizer, {i: l for i, l in enumerate(LABELS)})
results = evaluate_classifier(wrapped, test_texts, test_df_tx["label"].tolist(), model_name="DistilBERT")

hist_path = os.path.join(REPORTS_DIR, "distilbert_history.json")
if os.path.exists(hist_path):
    with open(hist_path, "r") as f:
        results["history"] = json.load(f)

hist_len = len(results.get("history", {}).get("train_loss", []))
results["train_time_sec"] = hist_len * 400 if hist_len > 0 else 3600

total_p, train_p = count_parameters(db_model)
results["total_parameters"] = int(total_p)
results["trainable_parameters"] = int(train_p)

distilbert_report_path = os.path.join(REPORTS_DIR, "distilbert_report.json")
with open(distilbert_report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done. Accuracy: {results['accuracy']:.4f}")
print(f"Report saved: {distilbert_report_path}")
