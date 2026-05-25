import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("[0] Starting resume test", flush=True)

import torch
print(f"[1] torch OK, CUDA: {torch.cuda.is_available()}", flush=True)

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
print("[2] transformers OK", flush=True)

from peft import LoraConfig, get_peft_model, TaskType
print("[3] peft OK", flush=True)

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
print("[4] Trainer OK", flush=True)

from src.config import SEED, DEVICE, MODELS_DIR, REPORTS_DIR, CHECKPOINTS_DIR, LABELS, NUM_CLASSES, BERT_MAX_LEN
print("[5] config OK", flush=True)

from src.config import set_seed
set_seed(SEED)
print("[6] seed set", flush=True)

from src.preprocessing import build_preprocessed_dataset
print("[7] preprocessing OK", flush=True)

from src.data_split import split_dataset
print("[8] data_split OK", flush=True)

print("[9] All imports successful", flush=True)
