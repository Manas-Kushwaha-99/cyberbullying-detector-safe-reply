import sys
sys.path.insert(0, '.')
from src.data_split import split_dataset
from src.preprocessing import build_preprocessed_dataset
from src.models.tfidf_svm import TfidfSvmClassifier
from src.evaluation.core_metrics import evaluate_classifier
from src.config import set_seed, SEED

print("Setting seed...")
set_seed(SEED)
print("Splitting dataset...")
train_df, val_df, test_df = split_dataset()
print(f"Split done: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

print("Preprocessing train...")
train_df = build_preprocessed_dataset(train_df, for_transformer=False)
print(f"Train preprocessed: {len(train_df)}")

print("Preprocessing val...")
val_df = build_preprocessed_dataset(val_df, for_transformer=False)
print(f"Val preprocessed: {len(val_df)}")

print("Preprocessing test...")
test_df = build_preprocessed_dataset(test_df, for_transformer=False)
print(f"Test preprocessed: {len(test_df)}")

X_train = train_df["processed_text"].tolist()
y_train = train_df["label"].tolist()
X_test = test_df["processed_text"].tolist()
y_test = test_df["label"].tolist()

print("Training SVM...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

clf = TfidfSvmClassifier()
clf.fit(X_train, y_train)
print("Training complete.")

print("Evaluating...")
results = evaluate_classifier(clf, X_test, y_test, model_name="TF-IDF+SVM")
print("Done.")
