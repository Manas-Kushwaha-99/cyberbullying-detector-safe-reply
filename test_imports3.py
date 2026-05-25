import sys
print("Python starting", flush=True)
print("Importing transformers module...", flush=True)
import transformers
print(f"Transformers version: {transformers.__version__}", flush=True)

print("Importing TrainingArguments...", flush=True)
from transformers import TrainingArguments
print("TrainingArguments OK", flush=True)

print("Importing Trainer (this may take a while)...", flush=True)
import time
start = time.time()
try:
    from transformers import Trainer
    elapsed = time.time() - start
    print(f"Trainer OK ({elapsed:.1f}s)", flush=True)
except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.1f}s: {e}", flush=True)
    import traceback
    traceback.print_exc()
