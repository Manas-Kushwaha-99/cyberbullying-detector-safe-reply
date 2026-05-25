"""Master pipeline: end-to-end execution of all experiments."""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.config import (
    SEED, DEVICE, DATA_PATH, MODELS_DIR, REPORTS_DIR, LABELS, NUM_CLASSES
)
from src.config import set_seed
from src.preprocessing import build_preprocessed_dataset
from src.data_split import split_dataset
from src.evaluation.core_metrics import evaluate_classifier
from src.evaluation.efficiency_benchmark import benchmark_inference, count_parameters
from src.evaluation.mcnemar_test import mcnemar_test
from src.evaluation.reply_eval import evaluate_replies_batch
from src.generation.safe_reply_generator import SafeReplyGenerator
from src.plotting.figure_generator import generate_all_figures
from src.tables.table_generator import generate_all_tables

warnings.filterwarnings("ignore")


def run_tfidf_svm(train_df, val_df, test_df):
    print("\n" + "=" * 70)
    print("Phase 1: TF-IDF + SVM")
    print("=" * 70)
    from src.training.train_tfidf_svm import train_tfidf_svm
    clf, results = train_tfidf_svm(csv_path=None, save_model=True)
    return clf, results


def run_lstm(train_df, val_df, test_df):
    print("\n" + "=" * 70)
    print("Phase 2: LSTM")
    print("=" * 70)
    from src.training.train_lstm import train_lstm
    model, results = train_lstm(csv_path=None, save_model=True)
    return model, results


def run_distilbert(train_df, val_df, test_df):
    print("\n" + "=" * 70)
    print("Phase 3: DistilBERT")
    print("=" * 70)
    from src.training.train_distilbert import train_distilbert
    model, results = train_distilbert(csv_path=None, save_model=True)
    return model, results


def run_distilbert_lora(train_df, val_df, test_df):
    print("\n" + "=" * 70)
    print("Phase 4: DistilBERT + LoRA")
    print("=" * 70)
    from src.training.train_distilbert_lora import train_distilbert_lora
    model, results = train_distilbert_lora(csv_path=None, save_model=True)
    return model, results


def run_efficiency_benchmark(models_dict, test_texts, test_df):
    print("\n" + "=" * 70)
    print("Phase 5: Efficiency Benchmarking")
    print("=" * 70)
    
    results = {}
    
    # TF-IDF + SVM
    if "tfidf_svm" in models_dict:
        print("[Benchmark] TF-IDF + SVM...")
        clf = models_dict["tfidf_svm"]
        bench = benchmark_inference(clf, test_texts, batch_size=256, device="cpu")
        results["tfidf_svm"] = bench
        
        # Update report
        report_path = os.path.join(REPORTS_DIR, "tfidf_svm_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            report["inference_time_sec"] = bench["inference_time_sec"]
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
    
    # LSTM
    if "lstm" in models_dict:
        print("[Benchmark] LSTM...")
        from src.models.lstm_model import LSTMWrapper, LSTM_MAX_LEN
        import pickle
        
        model = models_dict["lstm"]
        vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        
        wrapped = LSTMWrapper(model, vocab, LABELS)
        bench = benchmark_inference(wrapped, test_texts, batch_size=256, device=str(DEVICE))
        results["lstm"] = bench
        
        report_path = os.path.join(REPORTS_DIR, "lstm_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            report["inference_time_sec"] = bench["inference_time_sec"]
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
    
    # DistilBERT
    if "distilbert" in models_dict:
        print("[Benchmark] DistilBERT...")
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        from src.models.transformer_wrappers import DistilBERTWrapper
        from src.config import BERT_MAX_LEN
        
        model_path = os.path.join(MODELS_DIR, "distilbert")
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        wrapped = DistilBERTWrapper(model, tokenizer, {i: l for i, l in enumerate(LABELS)})
        
        bench = benchmark_inference(wrapped, test_texts, batch_size=32, device=str(DEVICE))
        results["distilbert"] = bench
        
        total_p, train_p = count_parameters(model)
        report_path = os.path.join(REPORTS_DIR, "distilbert_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            report["inference_time_sec"] = bench["inference_time_sec"]
            report["total_parameters"] = int(total_p)
            report["trainable_parameters"] = int(train_p)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
    
    # DistilBERT + LoRA
    if "distilbert_lora" in models_dict:
        print("[Benchmark] DistilBERT + LoRA...")
        from transformers import DistilBertTokenizerFast
        from peft import PeftModel
        from src.models.transformer_wrappers import DistilBERTLoRAWrapper
        from src.config import BERT_MODEL_NAME
        
        model_path = os.path.join(MODELS_DIR, "distilbert_lora")
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(DEVICE)
        wrapped = DistilBERTLoRAWrapper(model, tokenizer, {i: l for i, l in enumerate(LABELS)})
        
        bench = benchmark_inference(wrapped, test_texts, batch_size=32, device=str(DEVICE))
        results["distilbert_lora"] = bench
        
        total_p, train_p = count_parameters(model)
        report_path = os.path.join(REPORTS_DIR, "distilbert_lora_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            report["inference_time_sec"] = bench["inference_time_sec"]
            report["total_parameters"] = int(total_p)
            report["trainable_parameters"] = int(train_p)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
    
    return results


def run_mcnemar_test(test_df, test_texts):
    print("\n" + "=" * 70)
    print("Phase 6: McNemar's Test")
    print("=" * 70)
    
    # Load DistilBERT predictions
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from peft import PeftModel
    from src.config import BERT_MODEL_NAME, BERT_MAX_LEN
    
    label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
    y_true = [label2idx[l] for l in test_df["label"].tolist()]
    
    # DistilBERT predictions
    db_path = os.path.join(MODELS_DIR, "distilbert")
    db_tokenizer = DistilBertTokenizerFast.from_pretrained(db_path)
    db_model = DistilBertForSequenceClassification.from_pretrained(db_path).to(DEVICE)
    db_model.eval()
    
    db_preds = []
    batch_size = 32
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i:i+batch_size]
        enc = db_tokenizer(batch, truncation=True, padding="max_length",
                           max_length=BERT_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            outputs = db_model(**enc)
        db_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    
    # DistilBERT+LoRA predictions
    dbl_path = os.path.join(MODELS_DIR, "distilbert_lora")
    dbl_tokenizer = DistilBertTokenizerFast.from_pretrained(dbl_path)
    base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
    dbl_model = PeftModel.from_pretrained(base_model, dbl_path).to(DEVICE)
    dbl_model.eval()
    
    dbl_preds = []
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i:i+batch_size]
        enc = dbl_tokenizer(batch, truncation=True, padding="max_length",
                            max_length=BERT_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            outputs = dbl_model(**enc)
        dbl_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    
    result = mcnemar_test(np.array(y_true), np.array(db_preds), np.array(dbl_preds))
    result["comparison"] = "DistilBERT vs DistilBERT+LoRA"
    
    print(f"[McNemar] χ²={result['statistic']:.4f}, p={result['p_value']:.4f}")
    print(f"[McNemar] {result['significance']}")
    
    path = os.path.join(REPORTS_DIR, "mcnemar_test.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Saved] {path}")
    
    return result


def run_reply_generation(test_df, n_samples=50):
    print("\n" + "=" * 70)
    print("Phase 7: Safe Reply Generation & Evaluation")
    print("=" * 70)
    
    # Sample cyberbullying texts (exclude not_cyberbullying)
    cb_df = test_df[test_df["label"] != "not_cyberbullying"].sample(
        min(n_samples, len(test_df[test_df["label"] != "not_cyberbullying"])),
        random_state=SEED
    )
    
    messages = cb_df["processed_text"].tolist()
    
    # Initialize generator
    generator = SafeReplyGenerator(prompt_idx=0)
    
    # Generate replies
    print(f"[Generation] Generating replies for {len(messages)} messages...")
    replies = generator.generate_batch(messages, batch_size=4)
    
    # Save sample outputs
    samples = []
    for msg, reply in zip(messages, replies):
        samples.append({"message": msg, "reply": reply})
    
    samples_path = os.path.join(REPORTS_DIR, "reply_samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Sample replies -> {samples_path}")
    
    # Evaluate
    print("[Evaluation] Evaluating reply safety...")
    eval_results = evaluate_replies_batch(messages, replies)
    
    # Aggregate
    avg_results = {
        "toxicity": float(np.mean([r["toxicity"] for r in eval_results])),
        "empathy": float(np.mean([r["empathy"] for r in eval_results])),
        "relevance": float(np.mean([r["relevance"] for r in eval_results])),
        "safety_score": float(np.mean([r["safety_score"] for r in eval_results]))
    }
    
    print(f"[Reply Eval] Avg Toxicity: {avg_results['toxicity']:.4f}")
    print(f"[Reply Eval] Avg Empathy: {avg_results['empathy']:.4f}")
    print(f"[Reply Eval] Avg Relevance: {avg_results['relevance']:.4f}")
    print(f"[Reply Eval] Avg Safety: {avg_results['safety_score']:.4f}")
    
    path = os.path.join(REPORTS_DIR, "reply_evaluation.json")
    with open(path, "w") as f:
        json.dump(avg_results, f, indent=2)
    print(f"[Saved] {path}")
    
    return avg_results, samples


def main():
    set_seed(SEED)
    
    print("=" * 70)
    print("Cyberbullying Detector + Safe Reply Generator")
    print("Master Pipeline Execution")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {SEED}")
    
    # Load data
    print("\n[Pipeline] Loading and splitting dataset...")
    train_df, val_df, test_df = split_dataset()
    
    # Preprocess for non-transformer models
    test_df_proc = build_preprocessed_dataset(test_df.copy(), for_transformer=False)
    test_texts = test_df_proc["processed_text"].tolist()
    
    models_dict = {}
    
    # Phase 1-4: Training
    try:
        clf, _ = run_tfidf_svm(train_df, val_df, test_df)
        models_dict["tfidf_svm"] = clf
    except Exception as e:
        print(f"[Error] TF-IDF+SVM: {e}")
    
    try:
        lstm_model, _ = run_lstm(train_df, val_df, test_df)
        models_dict["lstm"] = lstm_model
    except Exception as e:
        print(f"[Error] LSTM: {e}")
    
    try:
        db_model, _ = run_distilbert(train_df, val_df, test_df)
        models_dict["distilbert"] = db_model
    except Exception as e:
        print(f"[Error] DistilBERT: {e}")
    
    try:
        dbl_model, _ = run_distilbert_lora(train_df, val_df, test_df)
        models_dict["distilbert_lora"] = dbl_model
    except Exception as e:
        print(f"[Error] DistilBERT+LoRA: {e}")
    
    # Phase 5: Efficiency Benchmarking
    try:
        if len(models_dict) >= 2:
            run_efficiency_benchmark(models_dict, test_texts, test_df)
    except Exception as e:
        print(f"[Error] Efficiency benchmarking: {e}")
    
    # Phase 6: McNemar Test
    try:
        if "distilbert" in models_dict and "distilbert_lora" in models_dict:
            run_mcnemar_test(test_df, test_df["processed_text"].tolist())
    except Exception as e:
        print(f"[Error] McNemar test: {e}")
    
    # Phase 7: Reply Generation
    try:
        test_df_transformer = build_preprocessed_dataset(test_df.copy(), for_transformer=True)
        run_reply_generation(test_df_transformer, n_samples=50)
    except Exception as e:
        print(f"[Error] Reply generation: {e}")
    
    # Phase 8: Figures & Tables
    print("\n" + "=" * 70)
    print("Phase 8: Generating Figures & Tables")
    print("=" * 70)
    
    try:
        generate_all_figures()
    except Exception as e:
        print(f"[Error] Figure generation: {e}")
    
    try:
        generate_all_tables()
    except Exception as e:
        print(f"[Error] Table generation: {e}")
    
    print("\n" + "=" * 70)
    print("Pipeline Execution Complete")
    print("=" * 70)
    print(f"Outputs saved to:")
    print(f"  - Models: {MODELS_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")
    print(f"  - Figures: {os.path.join(os.path.dirname(__file__), 'figures')}")
    print(f"  - Tables: {os.path.join(os.path.dirname(__file__), 'tables')}")


if __name__ == "__main__":
    main()
