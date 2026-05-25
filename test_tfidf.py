import sys
sys.path.insert(0, '.')
from src.training.train_tfidf_svm import train_tfidf_svm
print("Starting TF-IDF+SVM...")
train_tfidf_svm()
print("Done.")
