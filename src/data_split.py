"""Stratified train/validation/test split with fixed random seed."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, DATA_PATH, REPORTS_DIR

def split_dataset(csv_path=None, train_ratio=TRAIN_RATIO,
                  val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                  seed=SEED, label_col="label", text_col="text",
                  save_dir=REPORTS_DIR):
    """
    Perform stratified 80/10/10 split.
    
    Returns:
        train_df, val_df, test_df
    """
    if csv_path is None:
        csv_path = DATA_PATH
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    df = pd.read_csv(csv_path)
    
    # First split: train vs temp (val+test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df, test_size=temp_ratio, random_state=seed, stratify=df[label_col]
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_adjusted),
        random_state=seed, stratify=temp_df[label_col]
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save split indices for reproducibility
    os.makedirs(save_dir, exist_ok=True)
    train_df[[text_col, label_col]].to_csv(
        os.path.join(save_dir, "train_split.csv"), index=False
    )
    val_df[[text_col, label_col]].to_csv(
        os.path.join(save_dir, "val_split.csv"), index=False
    )
    test_df[[text_col, label_col]].to_csv(
        os.path.join(save_dir, "test_split.csv"), index=False
    )
    
    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"[Split] Train class distribution:")
    print(train_df[label_col].value_counts())
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    from src.config import set_seed
    set_seed()
    train_df, val_df, test_df = split_dataset()
