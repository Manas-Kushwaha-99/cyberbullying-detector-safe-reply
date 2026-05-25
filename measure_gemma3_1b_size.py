"""Measure exact VRAM usage of Gemma 3 1B with 4-bit quantization."""
import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Gemma 3 1B 4-bit Quantization - Exact Size Measurement")
print("=" * 70)

model_name = "google/gemma-3-1b-it"
print(f"\nLoading {model_name} in 4-bit...")

# Clear GPU cache before measurement
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Measure VRAM before loading
vram_before = torch.cuda.memory_allocated() / (1024 ** 2)
print(f"VRAM before loading: {vram_before:.2f} MB")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Measure VRAM after loading
vram_after = torch.cuda.memory_allocated() / (1024 ** 2)
vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
model_vram = vram_after - vram_before

print(f"\n" + "=" * 70)
print("VRAM USAGE RESULTS")
print("=" * 70)
print(f"VRAM before loading:     {vram_before:.2f} MB")
print(f"VRAM after loading:      {vram_after:.2f} MB")
print(f"VRAM used by model:      {model_vram:.2f} MB")
print(f"Peak VRAM during load:   {vram_peak:.2f} MB")

# Also check disk cache size
import glob
cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-1b-it")
if os.path.exists(cache_path):
    total_size = sum(os.path.getsize(f) for f in glob.glob(cache_path + "/**/*", recursive=True) if os.path.isfile(f))
    print(f"\nDisk cache size:         {total_size / (1024*1024):.2f} MB")
    print(f"Disk cache size:         {total_size / (1024*1024*1024):.2f} GB")

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters:        {total_params:,}")
print(f"Trainable parameters:    {trainable_params:,}")
print(f"Model size (fp32):       {total_params * 4 / (1024**2):.2f} MB (theoretical)")
print(f"Model size (4-bit):      {total_params * 0.5 / (1024**2):.2f} MB (theoretical)")
print(f"Actual VRAM usage:       {model_vram:.2f} MB")
