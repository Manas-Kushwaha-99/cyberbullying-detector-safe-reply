import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print("Importing split_dataset...")
from src.data_split import split_dataset
print("Calling split_dataset...")
train, val, test = split_dataset()
print(f"Done: {len(train)} {len(val)} {len(test)}")
