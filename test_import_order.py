import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("[0] Starting", flush=True)

import torch
print(f"[1] torch OK", flush=True)

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
print("[2] transformers pt1 OK", flush=True)

s = time.time()
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
print(f"[3] Trainer OK ({time.time()-s:.1f}s)", flush=True)

s = time.time()
from peft import LoraConfig, get_peft_model, TaskType
print(f"[4] peft OK ({time.time()-s:.1f}s)", flush=True)

print("[DONE]", flush=True)
