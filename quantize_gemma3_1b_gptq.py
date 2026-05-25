"""Quantize Gemma 3 1B to GPTQ 4-bit to reduce disk size to ~350 MB."""
import os, sys, glob, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("models/gemma3_1b_gptq", exist_ok=True)

print("=" * 70)
print("Gemma 3 1B GPTQ 4-bit Quantization")
print("=" * 70)

model_name = "google/gemma-3-1b-it"
print(f"\nLoading {model_name}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model (full precision first)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Prepare calibration data (simple examples)
calibration_data = [
    "Please be respectful to others online.",
    "Everyone deserves dignity regardless of their background.",
    "Let's keep conversations kind and constructive.",
    "Harassment is never acceptable.",
    "We should treat others the way we want to be treated.",
] * 20  # Repeat to have enough calibration data

# Quantization config for true INT4
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    model_file_base_name="gemma3_1b_4bit",
)

print("\nInitializing GPTQ quantizer...")
# Convert to GPTQ model
model_gptq = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

print("Quantizing to 4-bit...")
model_gptq.quantize(calibration_data, tokenizer)

print("\nSaving quantized model to models/gemma3_1b_gptq/ ...")
model_gptq.save_quantized("models/gemma3_1b_gptq", use_safetensors=True)
tokenizer.save_pretrained("models/gemma3_1b_gptq")

print("\n" + "=" * 70)
print("EXACT DISK SIZE MEASUREMENT")
print("=" * 70)

files = glob.glob("models/gemma3_1b_gptq/**/*", recursive=True)
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

# Compare with original
orig_cache = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-1b-it")
if os.path.exists(orig_cache):
    orig_size = sum(os.path.getsize(f) for f in glob.glob(orig_cache + "/**/*", recursive=True) if os.path.isfile(f))
    print(f"\nOriginal HF cache size:  {orig_size/(1024*1024):.2f} MB")
    print(f"GPTQ quantized size:     {total_size/(1024*1024):.2f} MB")
    print(f"Reduction:               {(1 - total_size/orig_size)*100:.1f}%")
