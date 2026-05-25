import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
start = time.time()
print("Importing reply_eval...", flush=True)
from src.evaluation.reply_eval import evaluate_replies_batch
elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s", flush=True)
