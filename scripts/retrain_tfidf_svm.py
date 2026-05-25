"""Retrain TF-IDF+SVM on enhanced dataset with correct import order."""
import os
import sys
import json
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

# CRITICAL: Import sklearn/pandas BEFORE torch to avoid Windows deadlock
from src.data_split import split_dataset
from src.preprocessing import build_preprocessed_dataset
from src.models.tfidf_svm import TfidfSvmClassifier
from src.evaluation.core_metrics import evaluate_classifier

# Now safe to import torch/config
from src.config import (
    SEED, MODELS_DIR, REPORTS_DIR,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, SVM_C, SVM_KERNEL,
    set_seed
)
set_seed(SEED)

print("=" * 60)
print("Training TF-IDF + SVM (Enhanced Dataset)")
print("=" * 60)

# Load & split data
train_df, val_df, test_df = split_dataset(csv_path='data/synthetic/cb_enhanced_v2.csv')

# Preprocess (non-transformer path)
train_df = build_preprocessed_dataset(train_df, for_transformer=False)
val_df = build_preprocessed_dataset(val_df, for_transformer=False)
test_df = build_preprocessed_dataset(test_df, for_transformer=False)

X_train = train_df["processed_text"].tolist()
y_train = train_df["label"].tolist()
X_test = test_df["processed_text"].tolist()
y_test = test_df["label"].tolist()

# Train
start_time = time.time()
clf = TfidfSvmClassifier(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
    C=SVM_C,
    kernel=SVM_KERNEL
)
clf.fit(X_train, y_train)
train_time = time.time() - start_time

# Evaluate
results = evaluate_classifier(clf, X_test, y_test, model_name="TF-IDF+SVM")
results["train_time_sec"] = train_time
results["inference_time_sec"] = None

# Save model
model_path = os.path.join(MODELS_DIR, "tfidf_svm.pkl")
clf.save(model_path)
print(f"[Saved] Model -> {model_path}")

# Save report
os.makedirs(REPORTS_DIR, exist_ok=True)
report_path = os.path.join(REPORTS_DIR, "tfidf_svm_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"[Saved] Report -> {report_path}")

print("\nTF-IDF+SVM retraining complete!")
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1: {results['f1_macro']:.4f}")
