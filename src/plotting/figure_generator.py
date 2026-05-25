"""Generate all 15 publication-quality figures."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from src.config import FIGURES_DIR, LABELS, REPORTS_DIR

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300

os.makedirs(FIGURES_DIR, exist_ok=True)


def save_figure(name, dpi=300, fmt="png"):
    path = os.path.join(FIGURES_DIR, f"{name}.{fmt}")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
    print(f"[Saved] Figure -> {path}")
    plt.close()


def load_report(model_name):
    path = os.path.join(REPORTS_DIR, f"{model_name}_report.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_history(model_name):
    path = os.path.join(REPORTS_DIR, f"{model_name}_history.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ── Figure 1: System Architecture Diagram ──────────────────────────
def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Figure 1: System Architecture", fontweight="bold", fontsize=32, pad=30)
    
    def box(x, y, w, h, text, color="lightblue", fontsize=20):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              edgecolor="black", facecolor=color, linewidth=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, wrap=True, fontweight="bold")
    
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=3))
    
    # Input
    box(4, 8.5, 2, 0.8, "Social Media Text", "#E8F4FD", fontsize=24)
    
    # Preprocessing
    box(3.5, 7, 3, 0.8, "Data Preprocessing Pipeline", "#D4EDDA", fontsize=24)
    arrow(5, 8.5, 5, 7.8)
    
    # Detection models
    box(0.5, 5, 2, 1, "TF-IDF + SVM", "#FFF3CD", fontsize=20)
    box(3, 5, 2, 1, "LSTM", "#FFF3CD", fontsize=20)
    box(5.5, 5, 2, 1, "DistilBERT", "#FFF3CD", fontsize=20)
    box(8, 5, 2, 1, "DistilBERT + LoRA", "#FFF3CD", fontsize=20)
    
    for x in [1.5, 4, 6.5, 9]:
        arrow(5, 7, x, 6)
    
    # Comparison & Selection
    box(3, 3, 4, 0.8, "Performance Comparison &\nBest Model Selection", "#F8D7DA", fontsize=20)
    for x in [1.5, 4, 6.5, 9]:
        arrow(x, 5, 5, 3.8)
    
    # Decision diamond (cyberbullying?)
    box(3, 1.5, 4, 0.8, "Cyberbullying Detected?", "#FFE5B4", fontsize=20)
    arrow(5, 3, 5, 2.3)
    
    # Yes branch → Safe Reply Generation
    arrow(5, 1.5, 7.5, 1.5)
    box(7.5, 1.2, 2.2, 0.6, "Safe Reply Generation\n(DistilGPT-2)", "#E2D4F0", fontsize=18)
    
    # No branch → No response
    arrow(5, 1.5, 2.5, 1.5)
    box(0.3, 1.2, 2.2, 0.6, "Not Cyberbullying\n(No Response)", "#F0F0F0", fontsize=18)
    
    # Output (from Safe Reply branch)
    arrow(8.6, 1.2, 8.6, 0.5)
    box(7.5, 0.2, 2.2, 0.6, "Safe, Empathetic Response", "#E8F4FD", fontsize=18)
    
    save_figure("01_system_architecture")
    plt.close()


# ── Figure 2: Dataset Class Distribution ───────────────────────────
def fig_class_distribution():
    counts = {
        "not_cyberbullying": 50000,
        "ethnicity/race": 17000,
        "gender/sexual": 17000,
        "religion": 15990
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor="black")
    ax.set_ylabel("Number of Samples")
    ax.set_xlabel("Class Label")
    ax.set_title("Figure 2: Dataset Class Distribution", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    
    save_figure("02_class_distribution")


# ── Figure 3: Accuracy Comparison ──────────────────────────────────
def fig_accuracy_comparison():
    models = ["TF-IDF+SVM", "LSTM", "DistilBERT", "DistilBERT+LoRA"]
    reports = [load_report("tfidf_svm"), load_report("lstm"),
               load_report("distilbert"), load_report("distilbert_lora")]
    accuracies = [r["accuracy"] if r else 0 for r in reports]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax.bar(models, accuracies, color=colors, edgecolor="black")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_title("Figure 3: Test Accuracy Comparison", fontweight="bold")
    ax.set_ylim(0, 1.05)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    
    save_figure("03_accuracy_comparison")


# ── Figure 4: Precision / Recall / F1 Comparison ──────────────────
def fig_precision_recall_f1():
    models = ["TF-IDF+SVM", "LSTM", "DistilBERT", "DistilBERT+LoRA"]
    reports = [load_report("tfidf_svm"), load_report("lstm"),
               load_report("distilbert"), load_report("distilbert_lora")]
    
    precision = [r["precision_macro"] if r else 0 for r in reports]
    recall = [r["recall_macro"] if r else 0 for r in reports]
    f1 = [r["f1_macro"] if r else 0 for r in reports]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#3498db", edgecolor="black")
    ax.bar(x, recall, width, label="Recall", color="#e74c3c", edgecolor="black")
    ax.bar(x + width, f1, width, label="F1-Score", color="#2ecc71", edgecolor="black")
    
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title("Figure 4: Precision / Recall / F1-Score Comparison (Macro)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    
    save_figure("04_precision_recall_f1")


# ── Figures 5-8: Confusion Matrices ────────────────────────────────
def fig_confusion_matrix(model_name, title_suffix, filename):
    report = load_report(model_name)
    if not report:
        print(f"[Skip] No report for {model_name}")
        return
    
    cm = np.array(report["confusion_matrix"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Figure {filename.split('_')[0]}: Confusion Matrix — {title_suffix}",
                 fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    
    save_figure(filename)


# ── Figures 9-11: Training vs Validation Loss Curves ──────────────
def fig_loss_curve(model_name, title_suffix, filename):
    history = load_history(model_name)
    if not history:
        print(f"[Skip] No history for {model_name}")
        return
    
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    
    if not train_loss or not val_loss:
        return
    
    epochs = range(1, len(train_loss) + 1)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, "b-o", label="Training Loss", markersize=5)
    ax.plot(epochs, val_loss, "r-s", label="Validation Loss", markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Figure {filename.split('_')[0]}: {title_suffix}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(filename)


# ── Figure 12: Training vs Validation Accuracy Curves ─────────────
def fig_accuracy_curves():
    """Generate unified training vs validation accuracy curves for all models."""
    models = [
        ("lstm", "LSTM"),
        ("distilbert", "DistilBERT"),
        ("distilbert_lora", "DistilBERT + LoRA")
    ]
    
    valid_models = []
    for hist_name, title in models:
        history = load_history(hist_name)
        if history and history.get("val_acc") and len(history["val_acc"]) > 0:
            valid_models.append((hist_name, title, history))
    
    if not valid_models:
        print("  [SKIP] No models have validation accuracy data")
        return
    
    n = len(valid_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, (hist_name, title, history) in zip(axes, valid_models):
        val_acc = history["val_acc"]
        epochs = range(1, len(val_acc) + 1)
        
        # Plot validation accuracy (all models have this)
        ax.plot(epochs, val_acc, "r-s", label="Val Acc", markersize=6, linewidth=2)
        
        # Plot training accuracy if available
        train_acc = history.get("train_acc", [])
        if train_acc and len(train_acc) > 0:
            # Ensure same length as epochs
            if len(train_acc) == len(val_acc):
                ax.plot(epochs, train_acc, "b-o", label="Train Acc", markersize=6, linewidth=2)
            elif len(train_acc) < len(val_acc):
                # Pad with None or truncate val_acc for plotting
                train_epochs = range(1, len(train_acc) + 1)
                ax.plot(train_epochs, train_acc, "b-o", label="Train Acc", markersize=6, linewidth=2)
        
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_ylim(0.95, 1.005)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
    
    fig.suptitle("Figure 12: Training vs Validation Accuracy Curves", 
                 fontweight="bold", fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure("12_accuracy_curves")
    plt.close()


# ── Figure 13: Training Time Comparison ────────────────────────────
def fig_training_time():
    models = ["TF-IDF+SVM", "LSTM", "DistilBERT", "DistilBERT+LoRA"]
    reports = [load_report("tfidf_svm"), load_report("lstm"),
               load_report("distilbert"), load_report("distilbert_lora")]
    times = [r["train_time_sec"] if r and "train_time_sec" in r else 0 for r in reports]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax.bar(models, times, color=colors, edgecolor="black")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_xlabel("Model")
    ax.set_title("Figure 13: Model Training Time Comparison", fontweight="bold")
    ax.set_yscale("log")
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    
    save_figure("13_training_time")


# ── Figure 14: VRAM / Parameter Efficiency ────────────────────────
def fig_efficiency():
    models = ["DistilBERT", "DistilBERT+LoRA"]
    reports = [load_report("distilbert"), load_report("distilbert_lora")]
    
    # Parameters
    total_params = []
    trainable_params = []
    for r in reports:
        if r:
            total_params.append(r.get("total_parameters", 0) / 1e6)
            trainable_params.append(r.get("trainable_parameters", 0) / 1e6)
        else:
            total_params.append(0)
            trainable_params.append(0)
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, total_params, width, label="Total Parameters", color="#3498db", edgecolor="black")
    ax.bar(x + width/2, trainable_params, width, label="Trainable Parameters", color="#e74c3c", edgecolor="black")
    
    ax.set_ylabel("Parameters (Millions)")
    ax.set_xlabel("Model")
    ax.set_title("Figure 14: Parameter Efficiency Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_yscale("log")
    
    for i, (t, tr) in enumerate(zip(total_params, trainable_params)):
        ax.text(i - width/2, t, f"{t:.1f}M", ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, tr, f"{tr:.1f}M", ha="center", va="bottom", fontsize=9)
    
    save_figure("14_parameter_efficiency")


# ── Figure 15: Reply Safety Score Comparison ──────────────────────
def fig_reply_safety():
    path = os.path.join(REPORTS_DIR, "reply_evaluation.json")
    if not os.path.exists(path):
        print("[Skip] No reply evaluation report found.")
        return
    
    with open(path, "r") as f:
        data = json.load(f)
    
    metrics = ["toxicity", "empathy", "relevance", "safety_score"]
    values = [data.get(m, 0) for m in metrics]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    bars = ax.bar([m.replace("_", " ").title() for m in metrics], values,
                  color=colors, edgecolor="black")
    ax.set_ylabel("Score")
    ax.set_title("Figure 15: Reply Safety Score Comparison", fontweight="bold")
    ax.set_ylim(0, 1.05)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    
    save_figure("15_reply_safety")


# ── Master Figure Generator ────────────────────────────────────────
def generate_all_figures():
    print("=" * 60)
    print("Generating All Figures")
    print("=" * 60)
    
    fig_system_architecture()
    fig_class_distribution()
    fig_accuracy_comparison()
    fig_precision_recall_f1()
    
    fig_confusion_matrix("tfidf_svm", "TF-IDF + SVM", "05_confusion_matrix_tfidf_svm")
    fig_confusion_matrix("lstm", "LSTM", "06_confusion_matrix_lstm")
    fig_confusion_matrix("distilbert", "DistilBERT", "07_confusion_matrix_distilbert")
    fig_confusion_matrix("distilbert_lora", "DistilBERT + LoRA", "08_confusion_matrix_distilbert_lora")
    
    fig_loss_curve("lstm", "LSTM Training vs Validation Loss", "09_loss_curve_lstm")
    fig_loss_curve("distilbert", "DistilBERT Training vs Validation Loss", "10_loss_curve_distilbert")
    fig_loss_curve("distilbert_lora", "LoRA Training vs Validation Loss", "11_loss_curve_distilbert_lora")
    
    fig_accuracy_curves()
    fig_training_time()
    fig_efficiency()
    fig_reply_safety()
    
    print("[Done] All figures generated.")


if __name__ == "__main__":
    generate_all_figures()
