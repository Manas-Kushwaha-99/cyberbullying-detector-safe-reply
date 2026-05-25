"""Quick cleanup script to fix label mappings and consolidate metrics."""
import os, sys, json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import REPORTS_DIR, TABLES_DIR, FIGURES_DIR, LABELS

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FIX RANDOM PREDICTIONS LABEL MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("1. Fixing Random Predictions Label Mapping")
print("="*60)

pred_path = os.path.join(REPORTS_DIR, "random_predictions.json")
if os.path.exists(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    fixed = []
    for s in samples:
        # Fix integer predictions
        pl = s["predicted_label"]
        try:
            idx = int(pl)
            if 0 <= idx < len(LABELS):
                s["predicted_label"] = LABELS[idx]
            else:
                s["predicted_label"] = str(pl)
        except (ValueError, TypeError):
            pass  # Already a string
        
        # Recompute correctness
        s["correct"] = (s["true_label"] == s["predicted_label"])
        fixed.append(s)
    
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, indent=2, ensure_ascii=False)
    
    # Regenerate markdown table
    df = pd.DataFrame(fixed)
    md_path = os.path.join(TABLES_DIR, "table_random_predictions.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Random Test Predictions (20 per model)\n\n")
        f.write(df.to_markdown(index=False))
    
    print(f"  Fixed {len(fixed)} prediction samples")
    print(f"  Saved -> {pred_path}")
    print(f"  Saved -> {md_path}")
else:
    print("  random_predictions.json not found, skipping")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD COMPLETE CONSOLIDATED METRICS TABLE (TRAIN/VAL/TEST)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. Building Complete Consolidated Metrics Table")
print("="*60)

# Load consolidated results (has train/val/test from compute_all_metrics.py)
consolidated_path = os.path.join(REPORTS_DIR, "consolidated_results.json")
if os.path.exists(consolidated_path):
    with open(consolidated_path, "r") as f:
        consolidated = json.load(f)
else:
    consolidated = {}

# Also load test reports for additional detail
model_reports = {
    "TF-IDF+SVM": "tfidf_svm_report.json",
    "LSTM": "lstm_report.json",
    "DistilBERT": "distilbert_report.json",
    "DistilBERT+LoRA": "distilbert_lora_report.json",
}

test_metrics = {}
for model_name, report_file in model_reports.items():
    rp = os.path.join(REPORTS_DIR, report_file)
    if os.path.exists(rp):
        with open(rp, "r") as f:
            data = json.load(f)
        test_metrics[model_name] = data

# Build comprehensive table
rows = []
for model_name in ["TF-IDF+SVM", "LSTM", "DistilBERT", "DistilBERT+LoRA"]:
    c = consolidated.get(model_name, {})
    t = test_metrics.get(model_name, {})
    
    row = {
        "Model": model_name,
        "Train Acc": f"{c.get('train_acc', 0):.4f}",
        "Train Precision": f"{c.get('train_precision', 0):.4f}" if c.get('train_precision') else "—",
        "Train Recall": f"{c.get('train_recall', 0):.4f}" if c.get('train_recall') else "—",
        "Train F1": f"{c.get('train_f1', 0):.4f}",
        "Val Acc": f"{c.get('val_acc', 0):.4f}",
        "Val Precision": f"{c.get('val_precision', 0):.4f}" if c.get('val_precision') else "—",
        "Val Recall": f"{c.get('val_recall', 0):.4f}" if c.get('val_recall') else "—",
        "Val F1": f"{c.get('val_f1', 0):.4f}",
        "Test Acc": f"{c.get('test_acc', 0):.4f}",
        "Test Precision": f"{t.get('precision_macro', c.get('test_precision', 0)):.4f}",
        "Test Recall": f"{t.get('recall_macro', c.get('test_recall', 0)):.4f}",
        "Test F1": f"{t.get('f1_macro', c.get('test_f1', 0)):.4f}",
    }
    rows.append(row)

df_metrics = pd.DataFrame(rows)
master_md = os.path.join(TABLES_DIR, "table_master_metrics.md")
with open(master_md, "w", encoding="utf-8") as f:
    f.write("# Master Metrics: Train / Validation / Test\n\n")
    f.write(df_metrics.to_markdown(index=False))
print(f"  Saved -> {master_md}")

# Also save as CSV
csv_path = os.path.join(TABLES_DIR, "table_master_metrics.csv")
df_metrics.to_csv(csv_path, index=False)
print(f"  Saved -> {csv_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. VERIFY ALL FIGURES AND TABLES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. Verifying All Figures and Tables")
print("="*60)

expected_figures = [
    "01_system_architecture.png",
    "02_class_distribution.png",
    "03_accuracy_comparison.png",
    "04_precision_recall_f1.png",
    "05_confusion_matrix_tfidf_svm.png",
    "06_confusion_matrix_lstm.png",
    "07_confusion_matrix_distilbert.png",
    "08_confusion_matrix_distilbert_lora.png",
    "09_loss_curve_lstm.png",
    "10_loss_curve_distilbert.png",
    "11_loss_curve_distilbert_lora.png",
    "12_accuracy_curves.png",
    "13_training_time.png",
    "14_parameter_efficiency.png",
    "15_reply_safety.png",
    "confusion_matrix_distilbert_lora_test.png",
    "confusion_matrix_distilbert_test.png",
    "confusion_matrix_lstm_test.png",
    "confusion_matrix_tf_idf_svm_test.png",
]

expected_tables = [
    "01_dataset_stats.md",
    "01_dataset_stats.csv",
    "01b_class_distribution.csv",
    "02_hyperparameters.md",
    "02_hyperparameters.csv",
    "03_performance_metrics.md",
    "03_performance_metrics.csv",
    "04_efficiency.md",
    "04_efficiency.csv",
    "05_reply_evaluation.md",
    "05_reply_evaluation.csv",
    "06_mcnemar.md",
    "06_mcnemar.csv",
    "consolidated_results.md",
    "table_full_metrics_test.md",
    "table_per_class_metrics_test.md",
    "table_random_predictions.md",
    "table_reply_verification.md",
    "table_master_metrics.md",
    "table_master_metrics.csv",
]

missing_figs = [f for f in expected_figures if not os.path.exists(os.path.join(FIGURES_DIR, f))]
missing_tabs = [t for t in expected_tables if not os.path.exists(os.path.join(TABLES_DIR, t))]

print(f"\n  Figures: {len(expected_figures) - len(missing_figs)}/{len(expected_figures)} present")
if missing_figs:
    print(f"  MISSING figures: {missing_figs}")
else:
    print(f"  All figures present.")

print(f"\n  Tables: {len(expected_tables) - len(missing_tabs)}/{len(expected_tables)} present")
if missing_tabs:
    print(f"  MISSING tables: {missing_tabs}")
else:
    print(f"  All tables present.")

# Save verification report
verification = {
    "figures": {
        "expected": len(expected_figures),
        "present": len(expected_figures) - len(missing_figs),
        "missing": missing_figs
    },
    "tables": {
        "expected": len(expected_tables),
        "present": len(expected_tables) - len(missing_tabs),
        "missing": missing_tabs
    },
    "status": "COMPLETE" if not missing_figs and not missing_tabs else "INCOMPLETE"
}

ver_path = os.path.join(REPORTS_DIR, "output_verification.json")
with open(ver_path, "w") as f:
    json.dump(verification, f, indent=2)
print(f"\n  Saved verification report -> {ver_path}")

print("\n" + "="*60)
print("CLEANUP COMPLETE")
print("="*60)
