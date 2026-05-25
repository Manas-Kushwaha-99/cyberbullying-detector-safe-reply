"""
Interactive Detection Model Tester
Loads all 4 retrained models and lets you test custom inputs.
"""
import os
import sys
import json
import pickle

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

# CRITICAL: Import sklearn/pandas BEFORE torch
from src.preprocessing import build_preprocessed_dataset
import pandas as pd

# Now safe to import torch
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from peft import PeftModel

from src.config import (
    DEVICE, MODELS_DIR, LSTM_MAX_LEN, LABELS, NUM_CLASSES,
    LSTM_EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_BIDIRECTIONAL,
    BERT_MAX_LEN, BERT_MODEL_NAME
)
from src.models.lstm_model import CyberbullyingLSTM
from src.models.transformer_wrappers import DistilBERTWrapper, DistilBERTLoRAWrapper
from src.models.tfidf_svm import TfidfSvmClassifier

print("=" * 70)
print("Loading Retrained Detection Models (Enhanced Dataset)")
print("=" * 70)

# ── Load TF-IDF + SVM ──────────────────────────────────────────────
print("[1/4] Loading TF-IDF + SVM...")
tfidf_clf = TfidfSvmClassifier.load(os.path.join(MODELS_DIR, "tfidf_svm.pkl"))
print("      OK")

# ── Load LSTM ─────────────────────────────────────────────────────
print("[2/4] Loading LSTM...")
vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
with open(vocab_path, "rb") as f:
    lstm_vocab = pickle.load(f)

lstm_model = CyberbullyingLSTM(
    vocab_size=len(lstm_vocab),
    embed_dim=LSTM_EMBED_DIM,
    hidden_dim=LSTM_HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    num_layers=LSTM_NUM_LAYERS,
    dropout=LSTM_DROPOUT,
    bidirectional=LSTM_BIDIRECTIONAL,
    pretrained_embeddings=None
).to(DEVICE)
lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), weights_only=True))
lstm_model.eval()
print("      OK")

# ── Load DistilBERT ───────────────────────────────────────────────
print("[3/4] Loading DistilBERT...")
distilbert_path = os.path.join(MODELS_DIR, "distilbert")
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_path)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path).to(DEVICE)
distilbert_model.eval()
distilbert_wrapper = DistilBERTWrapper(distilbert_model, distilbert_tokenizer, {i: l for i, l in enumerate(LABELS)})
print("      OK")

# ── Load DistilBERT + LoRA ────────────────────────────────────────
print("[4/4] Loading DistilBERT + LoRA...")
distilbert_lora_path = os.path.join(MODELS_DIR, "distilbert_lora")
distilbert_lora_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_lora_path)
base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
distilbert_lora_model = PeftModel.from_pretrained(base_model, distilbert_lora_path)
distilbert_lora_model = distilbert_lora_model.to(DEVICE)
distilbert_lora_model.eval()
distilbert_lora_wrapper = DistilBERTLoRAWrapper(distilbert_lora_model, distilbert_lora_tokenizer, {i: l for i, l in enumerate(LABELS)})
print("      OK")

print("\n" + "=" * 70)
print("All models loaded successfully!")
print("=" * 70)

# Predefined test examples (user's failure cases)
PREDEFINED_TESTS = [
    # Ethnicity/Race
    ("Go back to where you came from, nobody wants you here.", "ethnicity/race"),
    ("People of your race are always ruining this country.", "ethnicity/race"),
    ("Your culture is completely inferior to ours.", "ethnicity/race"),
    ("People from your ethnicity are always so backward.", "ethnicity/race"),
    ("Go back to where your race belongs.", "ethnicity/race"),
    # Gender/Sexual
    ("Go back to the kitchen and leave the real work to the men.", "gender/sexual"),
    ("Women are way too emotional to ever hold a leadership position.", "gender/sexual"),
    ("People with your sexual orientation are completely unnatural and wrong.", "gender/sexual"),
    ("Girls can't handle technical work.", "gender/sexual"),
    ("Your gender makes you less capable.", "gender/sexual"),
    # Religion
    ("Your religion is a violent cult that needs to be banned.", "religion"),
    ("All people who practice your faith are completely brainwashed.", "religion"),
    ("Your holy book is full of lies and promotes hatred.", "religion"),
    ("Your religion teaches strange things.", "religion"),
    ("No one wants your religion around.", "religion"),
    # Not Cyberbullying (challenging)
    ("I completely disagree with your political stance on this issue.", "not_cyberbullying"),
    ("That was a terrible movie and the plot made absolutely no sense.", "not_cyberbullying"),
    ("I hate how slow this computer is getting.", "not_cyberbullying"),
    ("The weather forecast says it might rain heavily this afternoon.", "not_cyberbullying"),
    ("Can someone help me fix this formatting error in my code?", "not_cyberbullying"),
    ("I disagree with your opinion.", "not_cyberbullying"),
    ("That joke was not funny.", "not_cyberbullying"),
    ("Let's discuss this calmly.", "not_cyberbullying"),
]

def predict_all(text):
    """Run text through all 4 models and return predictions."""
    # Preprocess for non-transformer
    df = pd.DataFrame({"text": [text], "label": ["not_cyberbullying"]})
    proc = build_preprocessed_dataset(df, for_transformer=False)
    proc_text = proc["processed_text"].iloc[0]
    
    # TF-IDF + SVM
    tfidf_pred = tfidf_clf.predict_labels([proc_text])[0]
    
    # LSTM
    tokens = proc_text.split()
    indices = [lstm_vocab.get(t, lstm_vocab["<UNK>"]) for t in tokens]
    if len(indices) < LSTM_MAX_LEN:
        indices += [lstm_vocab["<PAD>"]] * (LSTM_MAX_LEN - len(indices))
    else:
        indices = indices[:LSTM_MAX_LEN]
    
    with torch.no_grad():
        lstm_input = torch.tensor([indices], dtype=torch.long).to(DEVICE)
        lstm_out = lstm_model(lstm_input)
        lstm_pred = LABELS[torch.argmax(lstm_out, 1).item()]
    
    # DistilBERT
    distilbert_pred_idx = distilbert_wrapper.predict([text])[0]
    distilbert_pred = distilbert_wrapper.inv_label_map.get(distilbert_pred_idx, distilbert_pred_idx)
    
    # DistilBERT + LoRA
    distilbert_lora_pred_idx = distilbert_lora_wrapper.predict([text])[0]
    distilbert_lora_pred = distilbert_lora_wrapper.inv_label_map.get(distilbert_lora_pred_idx, distilbert_lora_pred_idx)
    
    return {
        "TF-IDF+SVM": tfidf_pred,
        "LSTM": lstm_pred,
        "DistilBERT": distilbert_pred,
        "DistilBERT+LoRA": distilbert_lora_pred,
    }

def run_predefined_tests():
    """Run all predefined tests and show results."""
    print("\n" + "=" * 70)
    print("Running Predefined Tests (User's Failure Examples)")
    print("=" * 70)
    
    correct_counts = {"TF-IDF+SVM": 0, "LSTM": 0, "DistilBERT": 0, "DistilBERT+LoRA": 0}
    total = len(PREDEFINED_TESTS)
    
    for text, true_label in PREDEFINED_TESTS:
        preds = predict_all(text)
        status = {}
        for model, pred in preds.items():
            is_correct = pred == true_label
            status[model] = "[OK]" if is_correct else "[FAIL]"
            if is_correct:
                correct_counts[model] += 1
        
        print(f"\nText: {text[:80]}...")
        print(f"True: {true_label}")
        for model, pred in preds.items():
            pred_str = str(pred)
            print(f"  {model:20s} -> {pred_str:20s} {status[model]}")
    
    print("\n" + "=" * 70)
    print("Predefined Test Summary")
    print("=" * 70)
    for model, correct in correct_counts.items():
        pct = (correct / total) * 100
        print(f"{model:20s}: {correct}/{total} correct ({pct:.1f}%)")
    
    return correct_counts

def interactive_mode():
    """Let user type custom texts."""
    print("\n" + "=" * 70)
    print("Interactive Mode")
    print("Type a message to classify (or 'quit' to exit)")
    print("=" * 70)
    
    while True:
        try:
            text = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not text or text.lower() in ("quit", "exit", "q"):
            print("Exiting...")
            break
        
        preds = predict_all(text)
        print(f"\nPredictions:")
        for model, pred in preds.items():
            print(f"  {model:20s} -> {pred}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode immediately")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        # Run predefined tests first, then offer interactive
        run_predefined_tests()
        print("\n" + "=" * 70)
        print("Predefined tests complete!")
        print("=" * 70)
        response = input("\nWould you like to enter interactive mode to test your own examples? (y/n): ")
        if response.lower() in ("y", "yes"):
            interactive_mode()
        else:
            print("Exiting. Run with -i flag to start interactive mode directly.")
