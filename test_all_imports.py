import sys
import os

print("[0] Starting", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print("[1] Path set", flush=True)

import torch
print(f"[2] torch imported, CUDA: {torch.cuda.is_available()}", flush=True)

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
print("[3] transformers imported", flush=True)

from peft import PeftModel
print("[4] peft imported", flush=True)

from src.config import SEED, DEVICE, MODELS_DIR, REPORTS_DIR, LABELS, NUM_CLASSES, BERT_MAX_LEN
print("[5] config imported", flush=True)

from src.config import set_seed
set_seed(SEED)
print("[6] seed set", flush=True)

from src.preprocessing import build_preprocessed_dataset
print("[7] preprocessing imported", flush=True)

from src.data_split import split_dataset
print("[8] data_split imported", flush=True)

from src.evaluation.core_metrics import evaluate_classifier
print("[9] core_metrics imported", flush=True)

from src.evaluation.efficiency_benchmark import benchmark_inference, count_parameters
print("[10] efficiency imported", flush=True)

from src.evaluation.mcnemar_test import mcnemar_test
print("[11] mcnemar imported", flush=True)

from src.evaluation.reply_eval import evaluate_replies_batch
print("[12] reply_eval imported", flush=True)

from src.generation.safe_reply_generator import SafeReplyGenerator
print("[13] generator imported", flush=True)

from src.models.transformer_wrappers import DistilBERTWrapper, DistilBERTLoRAWrapper
print("[14] wrappers imported", flush=True)

from src.plotting.figure_generator import generate_all_figures
print("[15] figures imported", flush=True)

from src.tables.table_generator import generate_all_tables
print("[16] tables imported", flush=True)

print("[DONE] All imports successful", flush=True)
