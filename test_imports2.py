import sys
print("Python starting", flush=True)
try:
    print("Importing transformers...", flush=True)
    from transformers import TrainingArguments
    print("TrainingArguments OK", flush=True)
    from transformers import Trainer
    print("Trainer OK", flush=True)
    print("Importing peft...", flush=True)
    from peft import LoraConfig
    print("LoraConfig OK", flush=True)
    print("Imports OK", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
