"""Training script for DistilBERT (full fine-tuning)."""
import os
import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

from src.config import (
    SEED, DEVICE, MODELS_DIR, CHECKPOINTS_DIR, REPORTS_DIR,
    BERT_MAX_LEN, BERT_BATCH_SIZE, BERT_LR, BERT_EPOCHS,
    BERT_WARMUP_RATIO, BERT_WEIGHT_DECAY, BERT_PATIENCE, BERT_MIN_DELTA,
    BERT_MODEL_NAME, LABELS, NUM_CLASSES
)
from src.preprocessing import build_preprocessed_dataset
from src.data_split import split_dataset
from src.evaluation.core_metrics import evaluate_classifier
from src.models.transformer_wrappers import DistilBERTWrapper


def compute_metrics(eval_pred):
    """Metrics function for HF Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


class DistilBERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def train_distilbert(csv_path=None, save_model=True):
    from src.config import set_seed
    set_seed(SEED)
    
    print("=" * 60)
    print("Training DistilBERT (Full Fine-tuning)")
    print("=" * 60)
    
    # Load & split data
    train_df, val_df, test_df = split_dataset(csv_path=csv_path)
    
    # Minimal preprocessing for transformer
    train_df = build_preprocessed_dataset(train_df, for_transformer=True)
    val_df = build_preprocessed_dataset(val_df, for_transformer=True)
    test_df = build_preprocessed_dataset(test_df, for_transformer=True)
    
    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    
    # Encode
    def encode(texts):
        return tokenizer(texts, truncation=True, padding="max_length",
                         max_length=BERT_MAX_LEN)
    
    label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
    
    train_enc = encode(train_df["processed_text"].tolist())
    val_enc = encode(val_df["processed_text"].tolist())
    test_enc = encode(test_df["processed_text"].tolist())
    
    train_labels = [label2idx[l] for l in train_df["label"].tolist()]
    val_labels = [label2idx[l] for l in val_df["label"].tolist()]
    test_labels = [label2idx[l] for l in test_df["label"].tolist()]
    
    train_dataset = DistilBERTDataset(train_enc, train_labels)
    val_dataset = DistilBERTDataset(val_enc, val_labels)
    test_dataset = DistilBERTDataset(test_enc, test_labels)
    
    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=NUM_CLASSES
    )
    
    # Training args
    training_args = TrainingArguments(
        output_dir=CHECKPOINTS_DIR,
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        learning_rate=BERT_LR,
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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=BERT_PATIENCE)]
    )
    
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    
    print(f"[DistilBERT] Training completed in {train_time:.2f}s")
    
    # Save model
    if save_model:
        model_path = os.path.join(MODELS_DIR, "distilbert")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"[Saved] Model -> {model_path}")
    
    # Extract history from trainer state
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    for log in trainer.state.log_history:
        if "loss" in log and "eval_loss" not in log:
            history["train_loss"].append(log["loss"])
        if "eval_loss" in log:
            history["val_loss"].append(log["eval_loss"])
        if "eval_accuracy" in log:
            history["val_acc"].append(log["eval_accuracy"])
    
    history_path = os.path.join(REPORTS_DIR, "distilbert_history.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Saved] History -> {history_path}")
    
    # Evaluate on test
    wrapped = DistilBERTWrapper(model, tokenizer, {i: l for i, l in enumerate(LABELS)})
    results = evaluate_classifier(wrapped, test_df["processed_text"].tolist(),
                                  test_df["label"].tolist(), model_name="DistilBERT")
    results["train_time_sec"] = float(train_time)
    results["history"] = history
    
    report_path = os.path.join(REPORTS_DIR, "distilbert_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Report -> {report_path}")
    
    return model, results


if __name__ == "__main__":
    train_distilbert()
