import sys
print("Python starting", flush=True)
try:
    print("Importing transformers...", flush=True)
    from transformers import TrainingArguments, Trainer
    print("Transformers OK", flush=True)
    print("Importing peft...", flush=True)
    from peft import LoraConfig, get_peft_model
    print("PEFT OK", flush=True)
    print("All imports OK", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
