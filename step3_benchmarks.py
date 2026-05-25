"""Step 3: Efficiency benchmarking + McNemar test."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Loading data...")
from src.data_split import split_dataset
from src.preprocessing import build_preprocessed_dataset
train_df, val_df, test_df = split_dataset()
test_df_tx = build_preprocessed_dataset(test_df.copy(), for_transformer=True)
test_texts = test_df_tx["processed_text"].tolist()

print("Benchmarking TF-IDF + SVM...")
from src.models.tfidf_svm import TfidfSvmClassifier
from src.evaluation.efficiency_benchmark import benchmark_inference

clf = TfidfSvmClassifier.load(os.path.join("models", "tfidf_svm.pkl"))
bench = benchmark_inference(clf, test_texts, batch_size=256, device="cpu")
report_path = os.path.join("reports", "tfidf_svm_report.json")
with open(report_path, "r") as f:
    report = json.load(f)
report["inference_time_sec"] = bench["inference_time_sec"]
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print("  TF-IDF+SVM done.")

print("Benchmarking LSTM...")
import torch, pickle
from src.models.lstm_model import CyberbullyingLSTM, LSTMWrapper
from src.config import DEVICE

checkpoint = torch.load(os.path.join("models", "lstm.pt"), map_location=DEVICE)
model = CyberbullyingLSTM(**checkpoint["config"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
with open(os.path.join("models", "lstm_vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)
wrapped = LSTMWrapper(model, vocab, ["not_cyberbullying", "ethnicity/race", "gender/sexual", "religion"])
bench = benchmark_inference(wrapped, test_texts, batch_size=256, device=str(DEVICE))
report_path = os.path.join("reports", "lstm_report.json")
with open(report_path, "r") as f:
    report = json.load(f)
report["inference_time_sec"] = bench["inference_time_sec"]
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print("  LSTM done.")

print("Benchmarking DistilBERT...")
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.models.transformer_wrappers import DistilBERTWrapper
from src.evaluation.efficiency_benchmark import count_parameters

db_tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join("models", "distilbert"))
db_model = DistilBertForSequenceClassification.from_pretrained(os.path.join("models", "distilbert")).to(DEVICE)
wrapped = DistilBERTWrapper(db_model, db_tokenizer, {i: l for i, l in enumerate(["not_cyberbullying", "ethnicity/race", "gender/sexual", "religion"])})
bench = benchmark_inference(wrapped, test_texts, batch_size=32, device=str(DEVICE))
total_p, train_p = count_parameters(db_model)
report_path = os.path.join("reports", "distilbert_report.json")
with open(report_path, "r") as f:
    report = json.load(f)
report["inference_time_sec"] = bench["inference_time_sec"]
report["total_parameters"] = int(total_p)
report["trainable_parameters"] = int(train_p)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("  DistilBERT done.")

print("Benchmarking DistilBERT + LoRA...")
from peft import PeftModel
from src.models.transformer_wrappers import DistilBERTLoRAWrapper

dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join("models", "distilbert_lora"))
base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
dbl_model = PeftModel.from_pretrained(base_model, os.path.join("models", "distilbert_lora")).to(DEVICE)
wrapped = DistilBERTLoRAWrapper(dbl_model, dbl_tokenizer, {i: l for i, l in enumerate(["not_cyberbullying", "ethnicity/race", "gender/sexual", "religion"])})
bench = benchmark_inference(wrapped, test_texts, batch_size=32, device=str(DEVICE))
total_p, train_p = count_parameters(dbl_model)
report_path = os.path.join("reports", "distilbert_lora_report.json")
with open(report_path, "r") as f:
    report = json.load(f)
report["inference_time_sec"] = bench["inference_time_sec"]
report["total_parameters"] = int(total_p)
report["trainable_parameters"] = int(train_p)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("  DistilBERT+LoRA done.")

print("\nMcNemar's Test...")
from src.evaluation.mcnemar_test import mcnemar_test
from src.config import BERT_MAX_LEN
import numpy as np

label2idx = {lbl: i for i, lbl in enumerate(["not_cyberbullying", "ethnicity/race", "gender/sexual", "religion"])}
y_true = [label2idx[l] for l in test_df_tx["label"].tolist()]

db_tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join("models", "distilbert"))
db_model = DistilBertForSequenceClassification.from_pretrained(os.path.join("models", "distilbert")).to(DEVICE)
db_model.eval()
db_preds = []
for i in range(0, len(test_texts), 32):
    batch = test_texts[i:i+32]
    enc = db_tokenizer(batch, truncation=True, padding="max_length", max_length=BERT_MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        outputs = db_model(**enc)
    db_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())

dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join("models", "distilbert_lora"))
base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
dbl_model = PeftModel.from_pretrained(base_model, os.path.join("models", "distilbert_lora")).to(DEVICE)
dbl_model.eval()
dbl_preds = []
for i in range(0, len(test_texts), 32):
    batch = test_texts[i:i+32]
    enc = dbl_tokenizer(batch, truncation=True, padding="max_length", max_length=BERT_MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        outputs = dbl_model(**enc)
    dbl_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())

result = mcnemar_test(np.array(y_true), np.array(db_preds), np.array(dbl_preds))
result["comparison"] = "DistilBERT vs DistilBERT+LoRA"

print(f"  chi2={result['statistic']:.4f}, p={result['p_value']:.4f}, {result['significance']}")

with open(os.path.join("reports", "mcnemar_test.json"), "w") as f:
    json.dump(result, f, indent=2)
print("  McNemar saved.")

print("\nAll steps complete!")
