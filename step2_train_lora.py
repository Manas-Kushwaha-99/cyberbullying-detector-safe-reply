"""Step 2: Train DistilBERT + LoRA."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Loading data...")
from src.data_split import split_dataset
train_df, val_df, test_df = split_dataset()

print("Preprocessing...")
from src.preprocessing import build_preprocessed_dataset
train_df_tx = build_preprocessed_dataset(train_df.copy(), for_transformer=True)
val_df_tx = build_preprocessed_dataset(val_df.copy(), for_transformer=True)
test_df_tx = build_preprocessed_dataset(test_df.copy(), for_transformer=True)

print("Importing ML libraries...")
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    SEED, DEVICE, MODELS_DIR, REPORTS_DIR, CHECKPOINTS_DIR,
    LABELS, NUM_CLASSES, BERT_MAX_LEN, LORA_BATCH_SIZE, LORA_LR,
    LORA_EPOCHS, LORA_PATIENCE, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, BERT_MODEL_NAME, BERT_WEIGHT_DECAY, BERT_WARMUP_RATIO
)
from src.config import set_seed
from src.evaluation.core_metrics import evaluate_classifier
from src.evaluation.efficiency_benchmark import count_parameters
from src.models.transformer_wrappers import DistilBERTLoRAWrapper

set_seed(SEED)
torch.cuda.empty_cache()

print("Tokenizing...")
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

print("Setting up DistilBERT + LoRA...")
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

print("Training started...")
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"Training completed in {train_time:.2f}s")

# Save model
dbl_model_path = os.path.join(MODELS_DIR, "distilbert_lora")
model.save_pretrained(dbl_model_path)
tokenizer.save_pretrained(dbl_model_path)
print(f"Model saved to {dbl_model_path}")

# Save history
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

# Evaluate
print("Evaluating on test set...")
test_texts = test_df_tx["processed_text"].tolist()
wrapped = DistilBERTLoRAWrapper(model, tokenizer, {i: l for i, l in enumerate(LABELS)})
results = evaluate_classifier(wrapped, test_texts, test_df_tx["label"].tolist(), model_name="DistilBERT+LoRA")
results["train_time_sec"] = float(train_time)
results["history"] = history

total_p, train_p = count_parameters(model)
results["total_parameters"] = int(total_p)
results["trainable_parameters"] = int(train_p)

dbl_report_path = os.path.join(REPORTS_DIR, "distilbert_lora_report.json")
with open(dbl_report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done. Accuracy: {results['accuracy']:.4f}")
print(f"Report saved: {dbl_report_path}")
