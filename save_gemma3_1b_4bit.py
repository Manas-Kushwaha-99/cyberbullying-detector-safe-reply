"""Save Gemma 3 1B in 4-bit quantized format to disk to reduce file size."""
import os, sys, glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("models/gemma3_1b_4bit", exist_ok=True)

print("=" * 70)
print("Saving Gemma 3 1B in 4-bit Quantized Format")
print("=" * 70)

model_name = "google/gemma-3-1b-it"
print(f"\nStep 1: Loading {model_name} with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

print("\nStep 2: Saving quantized model to models/gemma3_1b_4bit/ ...")
model.save_pretrained("models/gemma3_1b_4bit")
tokenizer.save_pretrained("models/gemma3_1b_4bit")

print("\nStep 3: Measuring exact disk size...")

# Calculate exact size
files = glob.glob("models/gemma3_1b_4bit/**/*", recursive=True)
file_sizes = {}
total_size = 0
for f in files:
    if os.path.isfile(f):
        size = os.path.getsize(f)
        fname = os.path.basename(f)
        total_size += size
        if fname in file_sizes:
            file_sizes[fname] += size
        else:
            file_sizes[fname] = size

print(f"\n{'File':<40} {'Size (MB)':>12}")
print("-" * 54)
for fname, size in sorted(file_sizes.items(), key=lambda x: x[1], reverse=True):
    print(f"{fname:<40} {size/(1024*1024):>12.2f}")
print("-" * 54)
print(f"{'TOTAL':<40} {total_size/(1024*1024):>12.2f} MB")
print(f"{'TOTAL':<40} {total_size/(1024*1024*1024):>12.2f} GB")

# Compare with original cache
import glob
original_cache = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-1b-it")
if os.path.exists(original_cache):
    orig_size = sum(os.path.getsize(f) for f in glob.glob(original_cache + "/**/*", recursive=True) if os.path.isfile(f))
    print(f"\nOriginal HF cache size: {orig_size/(1024*1024):.2f} MB")
    print(f"Saved quantized size:   {total_size/(1024*1024):.2f} MB")
    print(f"Reduction:            {(1 - total_size/orig_size)*100:.1f}%")
