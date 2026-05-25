"""
Fine-tune flan-t5-small on synthetic reply generation data.
Input: cyberbullying text
Target: safe reply
"""
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

import pandas as pd
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)

from src.config import SEED, DEVICE, MODELS_DIR, set_seed
set_seed(SEED)

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 128
BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 20
PATIENCE = 3

print("=" * 60)
print("Fine-tuning FLAN-T5-small for Safe Reply Generation")
print("=" * 60)

# Load synthetic data
df = pd.read_csv("data/synthetic/reply_synthetic.csv")
print(f"Loaded {len(df)} synthetic pairs")

# Split train/val (90/10)
train_df = df.sample(frac=0.9, random_state=SEED)
val_df = df.drop(train_df.index)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Format inputs with prefix
prefix = "generate safe reply: "

def preprocess(data):
    inputs = [prefix + text for text in data["cyberbullying_text"].tolist()]
    targets = data["safe_reply"].tolist()
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_TARGET_LEN, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_enc = preprocess(train_df)
val_enc = preprocess(val_df)

class ReplyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = ReplyDataset(train_enc)
val_dataset = ReplyDataset(val_enc)

# Model
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Training args
args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(MODELS_DIR, "flan_t5_small_reply_checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=SEED,
    report_to="none",
    disable_tqdm=False,
    predict_with_generate=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
)

print("Starting training...")
trainer.train()

# Save final model
save_path = os.path.join(MODELS_DIR, "flan_t5_small_reply")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n[Saved] Model -> {save_path}")

# Quick test on a few examples
print("\nQuick validation:")
test_texts = [
    "Go back to where you came from, nobody wants you here.",
    "Women are way too emotional to ever hold a leadership position.",
    "Your religion is a violent cult that needs to be banned.",
]

model.eval()
for text in test_texts:
    input_text = prefix + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_TARGET_LEN, num_beams=4, early_stopping=True)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input:  {text}")
    print(f"Reply:  {reply}")
    print()

print("FLAN-T5-small fine-tuning complete!")
