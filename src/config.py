"""Central configuration for reproducibility and hyperparameters."""
import os
import random
import numpy as np
import torch

# ── Reproducibility ───────────────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use enhanced dataset V2 with synthetic samples for retraining
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic", "cb_enhanced_v2.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models(New)")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
FIGURES_DIR = os.path.join(BASE_DIR, "figures(New)")
TABLES_DIR = os.path.join(BASE_DIR, "tables(New)")
REPORTS_DIR = os.path.join(BASE_DIR, "reports(New)")

for d in [MODELS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────
LABELS = ["not_cyberbullying", "ethnicity/race", "gender/sexual", "religion"]
NUM_CLASSES = len(LABELS)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ── Device ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Hyperparameters ─────────────────────────────────────────
# TF-IDF + SVM
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
SVM_C = 1.0
SVM_KERNEL = "linear"

# LSTM
LSTM_EMBED_DIM = 100
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.5
LSTM_BIDIRECTIONAL = True
LSTM_MAX_LEN = 128
LSTM_BATCH_SIZE = 256
LSTM_LR = 1e-3
LSTM_EPOCHS = 30
LSTM_PATIENCE = 3
LSTM_MIN_DELTA = 1e-4
LSTM_GLOVE_PATH = os.path.join(BASE_DIR, "glove.6B.100d.txt")

# DistilBERT
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 32
BERT_LR = 2e-5
BERT_EPOCHS = 10
BERT_WARMUP_RATIO = 0.1
BERT_WEIGHT_DECAY = 0.01
BERT_PATIENCE = 3
BERT_MIN_DELTA = 1e-4
BERT_MODEL_NAME = "distilbert-base-uncased"

# DistilBERT + LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_lin", "v_lin"]
LORA_BATCH_SIZE = 32
LORA_LR = 2e-4
LORA_EPOCHS = 10
LORA_PATIENCE = 3
LORA_MIN_DELTA = 1e-4

# Reply Generation
REPLY_MAX_LEN = 50
REPLY_TEMPERATURE = 0.7
REPLY_TOP_P = 0.9
REPLY_TOP_K = 50
REPLY_MODEL_NAME = "distilgpt2"

# ── Plotting ──────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# ── Empathy Lexicon (lightweight) ─────────────────────────────────
EMPATHY_KEYWORDS = {
    "empathy", "understand", "sorry", "hear", "feel", "care", "support",
    "help", "listen", "valid", "matter", "here", "you", "safe", "comfort",
    "hug", "love", "kind", "gentle", "peace", "calm", "breath", "relax",
    "hope", "better", "strong", "brave", "courage", "worth", "value",
    "respect", "accept", "welcome", "friend", "together", "not alone",
    "reach out", "talk", "share", "express", "emotion", "heart", "warm"
}

ENCOURAGEMENT_KEYWORDS = {
    "can", "will", "able", "capable", "potential", "grow", "learn", "improve",
    "progress", "forward", "move", "keep going", "don't give up", "persist",
    "resilient", "bounce back", "overcome", "achieve", "succeed", "win",
    "positive", "optimistic", "bright", "future", "tomorrow", "new day",
    "opportunity", "chance", "possibility", "dream", "goal", "aim", "strive"
}

CALMING_KEYWORDS = {
    "calm", "peaceful", "quiet", "still", "slow", "breathe", "deep breath",
    "relax", "unwind", "let go", "release", "soft", "gentle", "easy",
    "smooth", "steady", "stable", "balanced", "centered", "grounded",
    "mindful", "present", "moment", "now", "pause", "rest", "sleep",
    "comfortable", "cozy", "warm", "safe space", "haven", "refuge"
}
