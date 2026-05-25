"""Test DistilGPT-2 with strong prompt on manually preprocessed original dataset."""
import os, sys, json
import pandas as pd
import numpy as np

# Fix Windows console Unicode
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.preprocessing import preprocess_text
from src.generation.safe_reply_generator import SafeReplyGenerator
from src.config import SEED

# Load original dataset
print("Loading original dataset...")
df = pd.read_csv("cb_multi_labeled_balanced.csv")

# Manually preprocess text
print("Manually preprocessing text...")
df["processed_text"] = df["text"].apply(lambda x: preprocess_text(x, for_transformer=True))

# Filter cyberbullying examples (exclude not_cyberbullying)
cb_df = df[df["label"] != "not_cyberbullying"].sample(10, random_state=SEED)

messages_raw = cb_df["text"].tolist()
messages_proc = cb_df["processed_text"].tolist()
true_labels = cb_df["label"].tolist()

print("=" * 70)
print("DistilGPT-2 Strong Prompt Test - 10 Cyberbullying Examples")
print("=" * 70)

# Use the NEW strong prompt (index 3)
generator = SafeReplyGenerator(prompt_idx=3)

# Generate replies using PREPROCESSED text as input
print("\nGenerating replies with STRONG prompt on preprocessed text...\n")
replies = generator.generate_batch(messages_proc, batch_size=2)

results = []
for i, (raw, proc, true_lbl, reply) in enumerate(zip(messages_raw, messages_proc, true_labels, replies)):
    raw_safe = raw.encode("utf-8", errors="replace").decode("utf-8")
    proc_safe = proc.encode("utf-8", errors="replace").decode("utf-8")
    reply_safe = reply.encode("utf-8", errors="replace").decode("utf-8")
    
    print(f"--- Example {i+1} ---")
    print(f"True Label:     {true_lbl}")
    print(f"Original Text:  {raw_safe[:120]}...")
    print(f"Preprocessed:   {proc_safe[:120]}...")
    print(f"Reply:          {reply_safe}")
    print()
    results.append({
        "index": i+1,
        "true_label": true_lbl,
        "original_text": raw,
        "preprocessed_text": proc,
        "reply": reply
    })

# Save results
out_path = os.path.join("reports", "distilgpt2_strong_prompt_10_examples.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved results to {out_path}")
