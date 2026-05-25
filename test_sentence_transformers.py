import sys, time
start = time.time()
print("Importing sentence_transformers...", flush=True)
from sentence_transformers import SentenceTransformer, util
elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s", flush=True)
