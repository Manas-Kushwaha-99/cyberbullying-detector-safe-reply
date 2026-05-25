"""Test FLAN-T5-small with strong prompt on manually preprocessed original dataset."""
import os, sys, json, re
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Fix Windows console Unicode
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.preprocessing import preprocess_text
from src.config import SEED, DEVICE

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
print("FLAN-T5-Small Strong Prompt Test - 10 Cyberbullying Examples")
print("=" * 70)

# Load FLAN-T5-small
model_name = "google/flan-t5-small"
print(f"\nLoading {model_name}...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
model.eval()

# Strong prompt for FLAN-T5 (instruction-tuned, so it follows instructions better)
def make_prompt(message):
    return (
        "You are a compassionate crisis counselor. "
        "A person received this hurtful message online. "
        "Write a brief, empathetic, calming reply that supports them. "
        "Keep it under 25 words. Do not repeat hate speech. Be warm and human.\n\n"
        f"Hurtful message: {message}\n\n"
        "Supportive reply:"
    )

def generate_reply(message):
    prompt = make_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )
    
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Stop at first newline if any
    reply = re.split(r"[\n\r]", reply)[0].strip()
    # Ensure ends with punctuation
    if reply and reply[-1] not in ".!?":
        last_punct = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
        if last_punct > 0:
            reply = reply[:last_punct+1]
    return reply if reply else "I'm sorry you experienced this. You are not alone."

print("\nGenerating replies with FLAN-T5-small...\n")
results = []
for i, (raw, proc, true_lbl) in enumerate(zip(messages_raw, messages_proc, true_labels)):
    reply = generate_reply(proc)
    
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
out_path = os.path.join("reports", "flan_t5_small_10_examples.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved results to {out_path}")
