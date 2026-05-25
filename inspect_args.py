from transformers import TrainingArguments
import inspect
sig = inspect.signature(TrainingArguments.__init__)
params = [p for p in sig.parameters.keys()]
for p in params:
    print(p)
