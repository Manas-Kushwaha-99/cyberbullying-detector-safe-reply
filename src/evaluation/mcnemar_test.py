"""McNemar's statistical test for paired classifiers."""
import numpy as np
from scipy.stats import chi2

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    Perform McNemar's test between two classifiers.
    
    Args:
        y_true: Ground truth labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.
    
    Returns:
        Dict with statistic, p_value, significance.
    """
    # Contingency table
    # b: A correct, B wrong
    # c: A wrong, B correct
    b = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    c = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    
    # McNemar statistic with continuity correction
    if b + c == 0:
        statistic = 0.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
    
    p_value = 1 - chi2.cdf(statistic, df=1)
    
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "b": int(b),
        "c": int(c),
        "significance": significance,
        "interpretation": (
            f"McNemar's test: χ²={statistic:.4f}, p={p_value:.4f}. "
            f"Difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} "
            f"at α=0.05."
        )
    }
