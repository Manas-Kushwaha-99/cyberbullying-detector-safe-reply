"""Test Gemma 3 1B Q2_K GGUF on 10 cyberbullying examples."""
import os, sys, json, re
import pandas as pd
from llama_cpp import Llama

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.preprocessing import preprocess_text
from src.config import SEED

print("Loading dataset...")
df = pd.read_csv("cb_multi_labeled_balanced.csv")
df["processed_text"] = df["text"].apply(lambda x: preprocess_text(x, for_transformer=True))
cb_df = df[df["label"] != "not_cyberbullying"].sample(10, random_state=SEED)

messages_raw = cb_df["text"].tolist()
messages_proc = cb_df["processed_text"].tolist()
true_labels = cb_df["label"].tolist()

print("=" * 70)
print("Gemma 3 1B Q2_K GGUF Test")
print("=" * 70)

model_path = "models/gemma-3-1b-it-Q2_K.gguf"
print(f"\nLoading {model_path}...")

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    verbose=False,
)

def generate_reply(message):
    prompt = (
        'Generate a short de-escalating response to the following harmful online message.\n\n'
        f'Message:\n"{message}"\n\n'
        'Supportive reply: '
    )
    output = llm(
        prompt,
        max_tokens=30,
        temperature=0.3,
        top_p=0.75,
        stop=["\n"],
    )
    reply = output["choices"][0]["text"].strip()
    if reply and reply[-1] not in ".!?":
        last_punct = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
        if last_punct > 0:
            reply = reply[:last_punct+1]
    return reply if reply else "I'm sorry you experienced this."

print("\nGenerating replies...\n")
results = []
for i, (raw, proc, true_lbl) in enumerate(zip(messages_raw, messages_proc, true_labels)):
    reply = generate_reply(proc)
    
    raw_safe = raw.encode("utf-8", errors="replace").decode("utf-8")
    proc_safe = proc.encode("utf-8", errors="replace").decode("utf-8")
    reply_safe = reply.encode("utf-8", errors="replace").decode("utf-8")
    
    print(f"--- Example {i+1} ---")
    print(f"True Label:     {true_lbl}")
    print(f"Original:       {raw_safe[:120]}...")
    print(f"Preprocessed:   {proc_safe[:120]}...")
    print(f"Reply:          {reply_safe}")
    print()
    results.append({"index": i+1, "true_label": true_lbl, "original_text": raw, "preprocessed_text": proc, "reply": reply})

out_path = os.path.join("reports", "gemma3_1b_q2k_10_examples.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved to {out_path}")
