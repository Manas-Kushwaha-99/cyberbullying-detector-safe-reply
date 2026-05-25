"""Generate 10 DistilGPT-2 safe reply examples."""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_split import split_dataset
from src.preprocessing import build_preprocessed_dataset
from src.generation.safe_reply_generator import SafeReplyGenerator

# Load test data and get cyberbullying examples
train_df, val_df, test_df = split_dataset()
test_df_tx = build_preprocessed_dataset(test_df.copy(), for_transformer=True)
cb_df = test_df_tx[test_df_tx["label"] != "not_cyberbullying"].sample(10, random_state=42)

messages = cb_df["processed_text"].tolist()
true_labels = cb_df["label"].tolist()

print("=" * 70)
print("DistilGPT-2 Safe Reply Generation — 10 Cyberbullying Examples")
print("=" * 70)

# Initialize generator
generator = SafeReplyGenerator(prompt_idx=0)

# Generate replies
replies = generator.generate_batch(messages, batch_size=2)

print()
for i, (msg, true_lbl, reply) in enumerate(zip(messages, true_labels, replies)):
    print(f"--- Example {i+1} ---")
    print(f"True Label: {true_lbl}")
    print(f"Input:  {msg}")
    print(f"Reply:  {reply}")
    print()

# Save to reports
samples = []
for msg, true_lbl, reply in zip(messages, true_labels, replies):
    samples.append({"message": msg, "true_label": true_lbl, "reply": reply})

out_path = os.path.join("reports", "distilgpt2_10_examples.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)
print(f"Saved to {out_path}")
