import sys, time
start = time.time()
print("Importing detoxify...", flush=True)
from detoxify import Detoxify
elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s", flush=True)
