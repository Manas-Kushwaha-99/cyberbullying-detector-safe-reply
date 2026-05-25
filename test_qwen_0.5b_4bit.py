"""Test Qwen 0.5B with 4-bit quantization."""
import os, sys, json, re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
print("Qwen 0.5B (4-bit) Simplified Prompt Test")
print("=" * 70)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"\nLoading {model_name} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

def make_prompt(message):
    return (
        'Generate a short de-escalating response to the following harmful online message.\n\n'
        f'Message:\n"{message}"\n\n'
        'Supportive reply: '
    )

def generate_reply(message):
    prompt = make_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            top_p=0.75,
            temperature=0.3,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if prompt in reply:
        reply = reply[len(prompt):].strip()
    reply = re.split(r"[\n\r]", reply)[0].strip()
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

# Check cache size
import glob
print("\n" + "="*70)
print("MODEL SIZE INFO")
print("="*70)
cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--qwen--qwen2.5-0.5b-instruct")
if os.path.exists(cache_path):
    total_size = sum(os.path.getsize(f) for f in glob.glob(cache_path + "/**/*", recursive=True) if os.path.isfile(f))
    print(f"Qwen 0.5B cache size: {total_size / (1024*1024):.2f} MB")
else:
    print("Cache path not found yet")

out_path = os.path.join("reports", "qwen_0.5b_4bit_simple_prompt_10_examples.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
