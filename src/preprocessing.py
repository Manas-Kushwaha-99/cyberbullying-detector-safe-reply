"""Complete NLP preprocessing pipeline."""
import re
import string
import emoji
from collections import Counter

def preprocess_text(text, for_transformer=False):
    """
    Preprocess raw text for cyberbullying detection.
    
    Args:
        text: Raw input string.
        for_transformer: If True, apply minimal preprocessing to preserve
                         tokenizer semantics (DistilBERT compatibility).
    
    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing (for non-transformer; transformers handle case)
    if not for_transformer:
        text = text.lower()
    
    # 2. URL removal
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.IGNORECASE)
    
    # 3. Mention removal (@user)
    text = re.sub(r"@\w+", "", text)
    
    # 4. Hashtag handling: strip # but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # 5. Emoji handling: convert to textual description or remove
    if for_transformer:
        # Demojize for transformers so meaning is preserved
        text = emoji.demojize(text, delimiters=(" ", " "))
    else:
        text = emoji.replace_emoji(text, replace="")
    
    # 6. Punctuation cleanup
    if not for_transformer:
        text = text.translate(str.maketrans("", "", string.punctuation))
    else:
        # Keep basic punctuation for transformers but normalize repeats
        text = re.sub(r"([!?]){2,}", r"\1", text)
        text = re.sub(r"\.{2,}", "...", text)
    
    # 7. Whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def build_preprocessed_dataset(df, text_col="text", label_col="label",
                                for_transformer=False, remove_duplicates=True,
                                remove_nulls=True):
    """
    Apply preprocessing to a pandas DataFrame.
    
    Args:
        df: Input DataFrame.
        text_col: Name of text column.
        label_col: Name of label column.
        for_transformer: Transformer-compatible preprocessing flag.
        remove_duplicates: Whether to drop duplicate texts.
        remove_nulls: Whether to drop rows with null text/label.
    
    Returns:
        Preprocessed DataFrame with new column `processed_text`.
    """
    import pandas as pd
    
    df = df.copy()
    
    # Null handling
    if remove_nulls:
        df = df.dropna(subset=[text_col, label_col])
    
    # Preprocess
    df["processed_text"] = df[text_col].apply(
        lambda x: preprocess_text(x, for_transformer=for_transformer)
    )
    
    # Remove empty strings after preprocessing
    df = df[df["processed_text"].str.len() > 0]
    
    # Duplicate removal
    if remove_duplicates:
        df = df.drop_duplicates(subset=["processed_text"])
    
    df = df.reset_index(drop=True)
    return df


def get_class_weights(labels, method="balanced"):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: Array-like of class labels.
        method: 'balanced' or 'sqrt'.
    
    Returns:
        Dictionary mapping class index to weight.
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {i: w for i, w in zip(classes, weights)}
