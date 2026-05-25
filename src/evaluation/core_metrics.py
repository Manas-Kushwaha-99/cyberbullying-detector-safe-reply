"""Core evaluation metrics for all classifiers."""
import json
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from src.config import LABELS

def evaluate_classifier(model, X_test, y_test, model_name="Model",
                        return_predictions=False):
    """
    Evaluate a classifier and return comprehensive metrics.
    
    Args:
        model: Classifier with predict() or predict_labels() method.
        X_test: Test inputs.
        y_test: True labels (strings).
        model_name: Name for logging.
        return_predictions: If True, also return y_true, y_pred.
    
    Returns:
        Dictionary of metrics.
    """
    print(f"\n[{model_name}] Evaluating...")
    
    # Inference time
    start = time.time()
    if hasattr(model, "predict_labels"):
        y_pred = model.predict_labels(X_test)
    else:
        y_pred = model.predict(X_test)
        # Convert indices to labels if needed
        if isinstance(y_pred[0], (int, np.integer)):
            if hasattr(model, "inv_label_map"):
                y_pred = [model.inv_label_map[p] for p in y_pred]
            else:
                y_pred = [LABELS[p] for p in y_pred]
    inference_time = time.time() - start
    
    # Ensure labels are strings
    y_true = [str(y) for y in y_test]
    y_pred = [str(y) for y in y_pred]
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(y_true, y_pred, labels=LABELS,
                                   output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, labels=LABELS,
                                       zero_division=0)
    
    results = {
        "model": model_name,
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classification_report_str": report_str,
        "inference_time_sec": float(inference_time),
        "num_samples": len(y_true)
    }
    
    print(f"[{model_name}] Accuracy: {acc:.4f}")
    print(f"[{model_name}] Precision (macro): {precision:.4f}")
    print(f"[{model_name}] Recall (macro): {recall:.4f}")
    print(f"[{model_name}] F1 (macro): {f1:.4f}")
    print(f"[{model_name}] F1 (weighted): {f1_weighted:.4f}")
    print(f"[{model_name}] Inference time: {inference_time:.4f}s")
    
    if return_predictions:
        return results, y_true, y_pred
    return results
