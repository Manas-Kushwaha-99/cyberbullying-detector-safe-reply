"""Overfitting analysis utilities."""
import os
import json
import numpy as np

def load_history(model_name):
    """Load training history JSON."""
    path = os.path.join("reports", f"{model_name}_history.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def detect_overfitting(history, threshold=0.1):
    """
    Detect overfitting based on train/val loss divergence.
    
    Args:
        history: Dict with train_loss, val_loss lists.
        threshold: Minimum gap to flag overfitting.
    
    Returns:
        overfit_epoch: First epoch where val_loss - train_loss > threshold.
        max_gap: Maximum train/val loss gap.
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    
    if not train_loss or not val_loss:
        return None, None
    
    gaps = [v - t for t, v in zip(train_loss, val_loss)]
    max_gap = max(gaps)
    
    overfit_epoch = None
    for i, gap in enumerate(gaps):
        if gap > threshold:
            overfit_epoch = i + 1
            break
    
    return overfit_epoch, max_gap
