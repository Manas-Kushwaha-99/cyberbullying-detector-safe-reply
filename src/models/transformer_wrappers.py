"""Transformer model wrappers for inference."""
import torch
from src.config import DEVICE, BERT_MAX_LEN

class DistilBERTWrapper:
    def __init__(self, model, tokenizer, label_map, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.inv_label_map = label_map
        self.batch_size = batch_size
        self.model.eval()
    
    def predict(self, texts):
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                enc = self.tokenizer(batch, truncation=True, padding="max_length",
                                     max_length=BERT_MAX_LEN, return_tensors="pt")
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                outputs = self.model(**enc)
                all_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
        return all_preds


class DistilBERTLoRAWrapper:
    def __init__(self, model, tokenizer, label_map, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.inv_label_map = label_map
        self.batch_size = batch_size
        self.model.eval()
    
    def predict(self, texts):
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                enc = self.tokenizer(batch, truncation=True, padding="max_length",
                                     max_length=BERT_MAX_LEN, return_tensors="pt")
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                outputs = self.model(**enc)
                all_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
        return all_preds
