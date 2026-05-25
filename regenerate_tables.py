"""Regenerate all metric tables from updated consolidated_results.json."""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import REPORTS_DIR, TABLES_DIR

with open(os.path.join(REPORTS_DIR, "consolidated_results.json"), "r") as f:
    consolidated = json.load(f)

rows = []
for model_name in ["TF-IDF+SVM", "LSTM", "DistilBERT", "DistilBERT+LoRA"]:
    c = consolidated[model_name]
    rows.append({
        "Model": model_name,
        "Train Acc": f"{c['train_acc']:.4f}",
        "Train Precision": f"{c['train_precision']:.4f}",
        "Train Recall": f"{c['train_recall']:.4f}",
        "Train F1": f"{c['train_f1']:.4f}",
        "Val Acc": f"{c['val_acc']:.4f}",
        "Val Precision": f"{c['val_precision']:.4f}",
        "Val Recall": f"{c['val_recall']:.4f}",
        "Val F1": f"{c['val_f1']:.4f}",
        "Test Acc": f"{c['test_acc']:.4f}",
        "Test Precision": f"{c['test_precision']:.4f}",
        "Test Recall": f"{c['test_recall']:.4f}",
        "Test F1": f"{c['test_f1']:.4f}",
    })

df = pd.DataFrame(rows)

# Update consolidated_results.md
md_path = os.path.join(TABLES_DIR, "consolidated_results.md")
with open(md_path, "w") as f:
    f.write("# Consolidated Results: Train / Validation / Test Metrics\n\n")
    f.write(df.to_markdown(index=False))
print(f"Updated -> {md_path}")

# Update table_master_metrics.md
master_md = os.path.join(TABLES_DIR, "table_master_metrics.md")
with open(master_md, "w") as f:
    f.write("# Master Metrics: Train / Validation / Test\n\n")
    f.write(df.to_markdown(index=False))
print(f"Updated -> {master_md}")

# Update CSV
csv_path = os.path.join(TABLES_DIR, "table_master_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"Updated -> {csv_path}")

print("\nAll tables regenerated successfully.")
