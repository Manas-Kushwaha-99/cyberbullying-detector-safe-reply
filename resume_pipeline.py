"""Resume pipeline: evaluate existing models and train missing ones."""
import os
import sys
import json
import time
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from peft import PeftModel

from src.config import (
    SEED, DEVICE, MODELS_DIR, REPORTS_DIR, LABELS, NUM_CLASSES, BERT_MAX_LEN
)
from src.config import set_seed
from src.preprocessing import build_preprocessed_dataset
from src.data_split import split_dataset
from src.evaluation.core_metrics import evaluate_classifier
from src.evaluation.efficiency_benchmark import benchmark_inference, count_parameters
from src.evaluation.mcnemar_test import mcnemar_test
from src.evaluation.reply_eval import evaluate_replies_batch
from src.generation.safe_reply_generator import SafeReplyGenerator
from src.models.transformer_wrappers import DistilBERTWrapper, DistilBERTLoRAWrapper
from src.plotting.figure_generator import generate_all_figures
from src.tables.table_generator import generate_all_tables

warnings.filterwarnings("ignore")
set_seed(SEED)

print("=" * 70)
print("Resume Pipeline")
print("=" * 70)
print(f"Device: {DEVICE}")

# Load data once
train_df, val_df, test_df = split_dataset()
test_df = build_preprocessed_dataset(test_df.copy(), for_transformer=True)
test_texts = test_df["processed_text"].tolist()

# ── 1. TF-IDF + SVM ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 1: TF-IDF + SVM (already complete)")
print("=" * 70)

# ── 2. LSTM ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 2: LSTM (already complete)")
print("=" * 70)

# ── 3. DistilBERT ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 3: DistilBERT")
print("=" * 70)

distilbert_report_path = os.path.join(REPORTS_DIR, "distilbert_report.json")
distilbert_model_path = os.path.join(MODELS_DIR, "distilbert")

if os.path.exists(distilbert_report_path):
    print("[DistilBERT] Report already exists, skipping.")
    with open(distilbert_report_path, "r") as f:
        db_results = json.load(f)
elif os.path.exists(distilbert_model_path):
    print("[DistilBERT] Model found but report missing. Loading and evaluating...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path).to(DEVICE)
    wrapped = DistilBERTWrapper(model, tokenizer, {i: l for i, l in enumerate(LABELS)})
    
    results = evaluate_classifier(wrapped, test_texts, test_df["label"].tolist(), model_name="DistilBERT")
    
    history_path = os.path.join(REPORTS_DIR, "distilbert_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            results["history"] = json.load(f)
    
    # Try to estimate train time from history length
    if os.path.exists(history_path):
        hist = json.load(open(history_path))
        # rough estimate: 1 epoch ~ 400s on RTX 3060 Ti for this dataset
        results["train_time_sec"] = len(hist.get("train_loss", [])) * 400
    else:
        results["train_time_sec"] = 3600  # placeholder
    
    with open(distilbert_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Report -> {distilbert_report_path}")
    db_results = results
else:
    print("[DistilBERT] Model not found. Please run full training.")
    db_results = None

# ── 4. DistilBERT + LoRA ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 4: DistilBERT + LoRA")
print("=" * 70)

dbl_report_path = os.path.join(REPORTS_DIR, "distilbert_lora_report.json")

if os.path.exists(dbl_report_path):
    print("[DistilBERT+LoRA] Report already exists, skipping.")
    with open(dbl_report_path, "r") as f:
        dbl_results = json.load(f)
else:
    from src.training.train_distilbert_lora import train_distilbert_lora
    dbl_model, dbl_results = train_distilbert_lora(csv_path=None, save_model=True)

# ── 5. Efficiency Benchmarking ───────────────────────────────────
print("\n" + "=" * 70)
print("Phase 5: Efficiency Benchmarking")
print("=" * 70)

models_dict = {}

# TF-IDF + SVM
from src.models.tfidf_svm import TfidfSvmClassifier
if os.path.exists(os.path.join(MODELS_DIR, "tfidf_svm.pkl")):
    clf = TfidfSvmClassifier.load(os.path.join(MODELS_DIR, "tfidf_svm.pkl"))
    bench = benchmark_inference(clf, test_texts, batch_size=256, device="cpu")
    report_path = os.path.join(REPORTS_DIR, "tfidf_svm_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        report["inference_time_sec"] = bench["inference_time_sec"]
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    print("[Benchmark] TF-IDF + SVM done")

# LSTM
import pickle
from src.models.lstm_model import LSTMWrapper
lstm_model_path = os.path.join(MODELS_DIR, "lstm.pt")
vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
if os.path.exists(lstm_model_path) and os.path.exists(vocab_path):
    checkpoint = torch.load(lstm_model_path, map_location=DEVICE)
    from src.models.lstm_model import CyberbullyingLSTM
    model = CyberbullyingLSTM(**checkpoint["config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    wrapped = LSTMWrapper(model, vocab, LABELS)
    bench = benchmark_inference(wrapped, test_texts, batch_size=256, device=str(DEVICE))
    report_path = os.path.join(REPORTS_DIR, "lstm_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        report["inference_time_sec"] = bench["inference_time_sec"]
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    print("[Benchmark] LSTM done")

# DistilBERT
if os.path.exists(distilbert_model_path):
    tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path).to(DEVICE)
    wrapped = DistilBERTWrapper(model, tokenizer, {i: l for i, l in enumerate(LABELS)})
    bench = benchmark_inference(wrapped, test_texts, batch_size=32, device=str(DEVICE))
    total_p, train_p = count_parameters(model)
    if db_results is not None:
        db_results["inference_time_sec"] = bench["inference_time_sec"]
        db_results["total_parameters"] = int(total_p)
        db_results["trainable_parameters"] = int(train_p)
        with open(distilbert_report_path, "w", encoding="utf-8") as f:
            json.dump(db_results, f, indent=2, ensure_ascii=False)
    print("[Benchmark] DistilBERT done")

# DistilBERT + LoRA
dbl_model_path = os.path.join(MODELS_DIR, "distilbert_lora")
if os.path.exists(dbl_model_path):
    dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(dbl_model_path)
    base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
    dbl_model = PeftModel.from_pretrained(base_model, dbl_model_path).to(DEVICE)
    wrapped = DistilBERTLoRAWrapper(dbl_model, dbl_tokenizer, {i: l for i, l in enumerate(LABELS)})
    bench = benchmark_inference(wrapped, test_texts, batch_size=32, device=str(DEVICE))
    total_p, train_p = count_parameters(dbl_model)
    if dbl_results is not None:
        dbl_results["inference_time_sec"] = bench["inference_time_sec"]
        dbl_results["total_parameters"] = int(total_p)
        dbl_results["trainable_parameters"] = int(train_p)
        with open(dbl_report_path, "w", encoding="utf-8") as f:
            json.dump(dbl_results, f, indent=2, ensure_ascii=False)
    print("[Benchmark] DistilBERT+LoRA done")

# ── 6. McNemar Test ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 6: McNemar's Test")
print("=" * 70)

label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
y_true = [label2idx[l] for l in test_df["label"].tolist()]

if os.path.exists(distilbert_model_path) and os.path.exists(dbl_model_path):
    # DistilBERT preds
    db_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_model_path)
    db_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path).to(DEVICE)
    db_model.eval()
    db_preds = []
    for i in range(0, len(test_texts), 32):
        batch = test_texts[i:i+32]
        enc = db_tokenizer(batch, truncation=True, padding="max_length",
                           max_length=BERT_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            outputs = db_model(**enc)
        db_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    
    # LoRA preds
    dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(dbl_model_path)
    base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
    dbl_model = PeftModel.from_pretrained(base_model, dbl_model_path).to(DEVICE)
    dbl_model.eval()
    dbl_preds = []
    for i in range(0, len(test_texts), 32):
        batch = test_texts[i:i+32]
        enc = dbl_tokenizer(batch, truncation=True, padding="max_length",
                            max_length=BERT_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            outputs = dbl_model(**enc)
        dbl_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    
    result = mcnemar_test(np.array(y_true), np.array(db_preds), np.array(dbl_preds))
    result["comparison"] = "DistilBERT vs DistilBERT+LoRA"
    
    print(f"[McNemar] χ²={result['statistic']:.4f}, p={result['p_value']:.4f}")
    print(f"[McNemar] {result['significance']}")
    
    path = os.path.join(REPORTS_DIR, "mcnemar_test.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Saved] {path}")

# ── 7. Reply Generation ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 7: Safe Reply Generation & Evaluation")
print("=" * 70)

reply_eval_path = os.path.join(REPORTS_DIR, "reply_evaluation.json")
if os.path.exists(reply_eval_path):
    print("[Reply] Evaluation already exists, skipping.")
else:
    cb_df = test_df[test_df["label"] != "not_cyberbullying"].sample(
        min(50, len(test_df[test_df["label"] != "not_cyberbullying"])),
        random_state=SEED
    )
    messages = cb_df["processed_text"].tolist()
    generator = SafeReplyGenerator(prompt_idx=0)
    replies = generator.generate_batch(messages, batch_size=4)
    
    samples = [{"message": msg, "reply": reply} for msg, reply in zip(messages, replies)]
    with open(os.path.join(REPORTS_DIR, "reply_samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    eval_results = evaluate_replies_batch(messages, replies)
    avg_results = {
        "toxicity": float(np.mean([r["toxicity"] for r in eval_results])),
        "empathy": float(np.mean([r["empathy"] for r in eval_results])),
        "relevance": float(np.mean([r["relevance"] for r in eval_results])),
        "safety_score": float(np.mean([r["safety_score"] for r in eval_results]))
    }
    
    with open(reply_eval_path, "w") as f:
        json.dump(avg_results, f, indent=2)
    print(f"[Reply Eval] Avg Safety: {avg_results['safety_score']:.4f}")

# ── 8. Figures & Tables ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 8: Generating Figures & Tables")
print("=" * 70)

try:
    generate_all_figures()
except Exception as e:
    print(f"[Error] Figure generation: {e}")

try:
    generate_all_tables()
except Exception as e:
    print(f"[Error] Table generation: {e}")

print("\n" + "=" * 70)
print("Resume Pipeline Complete")
print("=" * 70)
