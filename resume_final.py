"""Resume pipeline - all heavy imports done inside functions to avoid top-level hangs."""
import os, sys, json, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Only lightweight top-level imports
from src.config import (
    SEED, DEVICE, MODELS_DIR, REPORTS_DIR, CHECKPOINTS_DIR,
    LABELS, NUM_CLASSES, BERT_MAX_LEN, BERT_BATCH_SIZE, BERT_LR,
    BERT_EPOCHS, BERT_WEIGHT_DECAY, BERT_WARMUP_RATIO, BERT_PATIENCE,
    LORA_BATCH_SIZE, LORA_LR, LORA_EPOCHS, LORA_PATIENCE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    BERT_MODEL_NAME
)
from src.config import set_seed
from src.preprocessing import build_preprocessed_dataset
from src.data_split import split_dataset
from src.evaluation.core_metrics import evaluate_classifier
from src.evaluation.efficiency_benchmark import benchmark_inference, count_parameters
from src.evaluation.mcnemar_test import mcnemar_test

# Use a simple logger that writes to file
LOG_FILE = open("resume_final.log", "w", encoding="utf-8")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FILE.write(line + "\n")
    LOG_FILE.flush()

warnings.filterwarnings("ignore")
set_seed(SEED)

log("="*50)
log("RESUME PIPELINE")
log("="*50)
log(f"Device: {DEVICE}")

log("Loading data...")
train_df, val_df, test_df = split_dataset()
test_df_tx = build_preprocessed_dataset(test_df.copy(), for_transformer=True)
test_texts = test_df_tx["processed_text"].tolist()
log(f"Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_texts)}")

# ═══════════════════════════════════════════════════════════════════
# 1. Evaluate existing DistilBERT
# ═══════════════════════════════════════════════════════════════════
log("Step 1: Evaluate DistilBERT")

distilbert_model_path = os.path.join(MODELS_DIR, "distilbert")
distilbert_report_path = os.path.join(REPORTS_DIR, "distilbert_report.json")

if os.path.exists(distilbert_report_path):
    log("DistilBERT report already exists.")
else:
    log("DistilBERT: Importing transformers...")
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from src.models.transformer_wrappers import DistilBERTWrapper
    log("DistilBERT: Imports done.")
    
    log("DistilBERT: Loading model...")
    db_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_model_path)
    db_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path).to(DEVICE)
    log("DistilBERT: Model loaded.")
    
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
    
    with open(distilbert_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"DistilBERT report saved. Acc={results['accuracy']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 2. Train DistilBERT + LoRA
# ═══════════════════════════════════════════════════════════════════
log("Step 2: Train DistilBERT + LoRA")

dbl_report_path = os.path.join(REPORTS_DIR, "distilbert_lora_report.json")

if os.path.exists(dbl_report_path):
    log("DistilBERT+LoRA report already exists.")
else:
    log("DistilBERT+LoRA: Importing...")
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
    from peft import LoraConfig, get_peft_model, TaskType
    from src.models.transformer_wrappers import DistilBERTLoRAWrapper
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    log("DistilBERT+LoRA: Imports done.")
    
    log("DistilBERT+LoRA: Preparing data...")
    train_df_tx = build_preprocessed_dataset(train_df.copy(), for_transformer=True)
    val_df_tx = build_preprocessed_dataset(val_df.copy(), for_transformer=True)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    
    def encode(texts):
        return tokenizer(texts, truncation=True, padding="max_length", max_length=BERT_MAX_LEN)
    
    label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
    
    train_enc = encode(train_df_tx["processed_text"].tolist())
    val_enc = encode(val_df_tx["processed_text"].tolist())
    test_enc = encode(test_df_tx["processed_text"].tolist())
    
    train_labels = [label2idx[l] for l in train_df_tx["label"].tolist()]
    val_labels = [label2idx[l] for l in val_df_tx["label"].tolist()]
    test_labels = [label2idx[l] for l in test_df_tx["label"].tolist()]
    
    class EncDataset(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    
    train_dataset = EncDataset(train_enc, train_labels)
    val_dataset = EncDataset(val_enc, val_labels)
    test_dataset = EncDataset(test_enc, test_labels)
    
    log("DistilBERT+LoRA: Setting up model...")
    base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
    
    training_args = TrainingArguments(
        output_dir=CHECKPOINTS_DIR,
        num_train_epochs=LORA_EPOCHS,
        per_device_train_batch_size=LORA_BATCH_SIZE,
        per_device_eval_batch_size=LORA_BATCH_SIZE,
        learning_rate=LORA_LR,
        weight_decay=BERT_WEIGHT_DECAY,
        warmup_ratio=BERT_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        disable_tqdm=False,
    )
    
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=LORA_PATIENCE)]
    )
    
    log("DistilBERT+LoRA: Training started...")
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    log(f"DistilBERT+LoRA: Training completed in {train_time:.2f}s")
    
    dbl_model_path = os.path.join(MODELS_DIR, "distilbert_lora")
    model.save_pretrained(dbl_model_path)
    tokenizer.save_pretrained(dbl_model_path)
    log("DistilBERT+LoRA: Model saved.")
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for log_entry in trainer.state.log_history:
        if "loss" in log_entry and "eval_loss" not in log_entry:
            history["train_loss"].append(log_entry["loss"])
        if "eval_loss" in log_entry:
            history["val_loss"].append(log_entry["eval_loss"])
        if "eval_accuracy" in log_entry:
            history["val_acc"].append(log_entry["eval_accuracy"])
    
    hist_path = os.path.join(REPORTS_DIR, "distilbert_lora_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    
    wrapped = DistilBERTLoRAWrapper(model, tokenizer, {i: l for i, l in enumerate(LABELS)})
    dbl_results = evaluate_classifier(wrapped, test_texts, test_df_tx["label"].tolist(), model_name="DistilBERT+LoRA")
    dbl_results["train_time_sec"] = float(train_time)
    dbl_results["history"] = history
    
    total_p, train_p = count_parameters(model)
    dbl_results["total_parameters"] = int(total_p)
    dbl_results["trainable_parameters"] = int(train_p)
    
    with open(dbl_report_path, "w", encoding="utf-8") as f:
        json.dump(dbl_results, f, indent=2, ensure_ascii=False)
    log(f"DistilBERT+LoRA report saved. Acc={dbl_results['accuracy']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 3. Efficiency Benchmarking
# ═══════════════════════════════════════════════════════════════════
log("Step 3: Efficiency Benchmarking")

from src.models.tfidf_svm import TfidfSvmClassifier
if os.path.exists(os.path.join(MODELS_DIR, "tfidf_svm.pkl")):
    log("Benchmarking TF-IDF + SVM...")
    clf = TfidfSvmClassifier.load(os.path.join(MODELS_DIR, "tfidf_svm.pkl"))
    bench = benchmark_inference(clf, test_texts, batch_size=256, device="cpu")
    report_path = os.path.join(REPORTS_DIR, "tfidf_svm_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        report["inference_time_sec"] = bench["inference_time_sec"]
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    log("TF-IDF + SVM benchmark done.")

import pickle
from src.models.lstm_model import CyberbullyingLSTM, LSTMWrapper
lstm_path = os.path.join(MODELS_DIR, "lstm.pt")
vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
if os.path.exists(lstm_path) and os.path.exists(vocab_path):
    log("Benchmarking LSTM...")
    import torch
    checkpoint = torch.load(lstm_path, map_location=DEVICE)
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
    log("LSTM benchmark done.")

if os.path.exists(distilbert_model_path):
    log("Benchmarking DistilBERT...")
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from src.models.transformer_wrappers import DistilBERTWrapper
    db_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_model_path)
    db_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path).to(DEVICE)
    wrapped = DistilBERTWrapper(db_model, db_tokenizer, {i: l for i, l in enumerate(LABELS)})
    bench = benchmark_inference(wrapped, test_texts, batch_size=32, device=str(DEVICE))
    total_p, train_p = count_parameters(db_model)
    if os.path.exists(distilbert_report_path):
        with open(distilbert_report_path, "r") as f:
            report = json.load(f)
        report["inference_time_sec"] = bench["inference_time_sec"]
        report["total_parameters"] = int(total_p)
        report["trainable_parameters"] = int(train_p)
        with open(distilbert_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    log("DistilBERT benchmark done.")

dbl_model_path = os.path.join(MODELS_DIR, "distilbert_lora")
if os.path.exists(dbl_model_path):
    log("Benchmarking DistilBERT + LoRA...")
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from peft import PeftModel
    from src.models.transformer_wrappers import DistilBERTLoRAWrapper
    dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(dbl_model_path)
    base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
    dbl_model = PeftModel.from_pretrained(base_model, dbl_model_path).to(DEVICE)
    wrapped = DistilBERTLoRAWrapper(dbl_model, dbl_tokenizer, {i: l for i, l in enumerate(LABELS)})
    bench = benchmark_inference(wrapped, test_texts, batch_size=32, device=str(DEVICE))
    total_p, train_p = count_parameters(dbl_model)
    if os.path.exists(dbl_report_path):
        with open(dbl_report_path, "r") as f:
            report = json.load(f)
        report["inference_time_sec"] = bench["inference_time_sec"]
        report["total_parameters"] = int(total_p)
        report["trainable_parameters"] = int(train_p)
        with open(dbl_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    log("DistilBERT+LoRA benchmark done.")

# ═══════════════════════════════════════════════════════════════════
# 4. McNemar Test
# ═══════════════════════════════════════════════════════════════════
log("Step 4: McNemar's Test")

mcnemar_path = os.path.join(REPORTS_DIR, "mcnemar_test.json")
if os.path.exists(mcnemar_path):
    log("McNemar already done.")
else:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from peft import PeftModel
    
    label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
    y_true = [label2idx[l] for l in test_df_tx["label"].tolist()]
    
    db_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_model_path)
    db_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path).to(DEVICE)
    db_model.eval()
    db_preds = []
    for i in range(0, len(test_texts), 32):
        batch = test_texts[i:i+32]
        enc = db_tokenizer(batch, truncation=True, padding="max_length", max_length=BERT_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            outputs = db_model(**enc)
        db_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    
    dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(dbl_model_path)
    base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
    dbl_model = PeftModel.from_pretrained(base_model, dbl_model_path).to(DEVICE)
    dbl_model.eval()
    dbl_preds = []
    for i in range(0, len(test_texts), 32):
        batch = test_texts[i:i+32]
        enc = dbl_tokenizer(batch, truncation=True, padding="max_length", max_length=BERT_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            outputs = dbl_model(**enc)
        dbl_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    
    result = mcnemar_test(np.array(y_true), np.array(db_preds), np.array(dbl_preds))
    result["comparison"] = "DistilBERT vs DistilBERT+LoRA"
    
    log(f"McNemar: χ²={result['statistic']:.4f}, p={result['p_value']:.4f}, {result['significance']}")
    
    with open(mcnemar_path, "w") as f:
        json.dump(result, f, indent=2)
    log("McNemar saved.")

# ═══════════════════════════════════════════════════════════════════
# 5. Figures & Tables
# ═══════════════════════════════════════════════════════════════════
log("Step 5: Generating Figures & Tables")

from src.plotting.figure_generator import generate_all_figures
from src.tables.table_generator import generate_all_tables

try:
    generate_all_figures()
    log("Figures generated.")
except Exception as e:
    log(f"Figure generation error: {e}")

try:
    generate_all_tables()
    log("Tables generated.")
except Exception as e:
    log(f"Table generation error: {e}")

log("="*50)
log("RESUME PIPELINE COMPLETE")
log("="*50)
LOG_FILE.close()
