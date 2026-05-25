"""TF-IDF + SVM model wrapper."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib

class TfidfSvmClassifier:
    def __init__(self, max_features=10000, ngram_range=(1, 2), C=1.0, kernel="linear"):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.kernel = kernel
        self.pipeline = None
        self.label_map = None
        self.inv_label_map = None
    
    def fit(self, texts, labels):
        """Fit the TF-IDF + SVM pipeline."""
        # Build label mapping
        unique_labels = sorted(set(labels))
        self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.inv_label_map = {i: lbl for lbl, i in self.label_map.items()}
        
        y = [self.label_map[l] for l in labels]
        
        # LinearSVC is much faster than SVC for large datasets
        base_svm = LinearSVC(C=self.C, class_weight="balanced", random_state=42,
                             max_iter=2000, dual="auto")
        
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                strip_accents="unicode",
                stop_words="english"
            )),
            ("svm", base_svm)
        ])
        
        self.pipeline.fit(texts, y)
        return self
    
    def predict(self, texts):
        """Return predicted class indices."""
        return self.pipeline.predict(texts)
    
    def predict_labels(self, texts):
        """Return predicted string labels."""
        preds = self.predict(texts)
        return [self.inv_label_map[p] for p in preds]
    
    def save(self, path):
        joblib.dump({
            "pipeline": self.pipeline,
            "label_map": self.label_map,
            "inv_label_map": self.inv_label_map,
            "config": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "C": self.C,
                "kernel": self.kernel
            }
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls(**data["config"])
        obj.pipeline = data["pipeline"]
        obj.label_map = data["label_map"]
        obj.inv_label_map = data["inv_label_map"]
        return obj
