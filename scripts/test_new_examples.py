"""Test new examples from user on all 4 retrained detection models."""
import os
import sys
import json
import pickle

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

from src.preprocessing import build_preprocessed_dataset
import pandas as pd
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

# Load all models
print("Loading models...")
tfidf_clf = TfidfSvmClassifier.load(os.path.join(MODELS_DIR, "tfidf_svm.pkl"))

with open(os.path.join(MODELS_DIR, "lstm_vocab.pkl"), "rb") as f:
    lstm_vocab = pickle.load(f)
lstm_model = CyberbullyingLSTM(
    vocab_size=len(lstm_vocab), embed_dim=LSTM_EMBED_DIM, hidden_dim=LSTM_HIDDEN_DIM,
    num_classes=NUM_CLASSES, num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT,
    bidirectional=LSTM_BIDIRECTIONAL, pretrained_embeddings=None
).to(DEVICE)
lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), weights_only=True, map_location='cpu'))
lstm_model.eval()

distilbert_path = os.path.join(MODELS_DIR, "distilbert")
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_path)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path).to(DEVICE)
distilbert_model.eval()
distilbert_wrapper = DistilBERTWrapper(distilbert_model, distilbert_tokenizer, {i: l for i, l in enumerate(LABELS)})

distilbert_lora_path = os.path.join(MODELS_DIR, "distilbert_lora")
distilbert_lora_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_lora_path)
base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
distilbert_lora_model = PeftModel.from_pretrained(base_model, distilbert_lora_path).to(DEVICE)
distilbert_lora_model.eval()
distilbert_lora_wrapper = DistilBERTLoRAWrapper(distilbert_lora_model, distilbert_lora_tokenizer, {i: l for i, l in enumerate(LABELS)})
print("All models loaded.\n")

# Test examples
TEST_EXAMPLES = [
    # Ethnicity/Race
    ("People from your race always ruin everything online.", "ethnicity/race"),
    ("Go back to your own country, nobody wants you here.", "ethnicity/race"),
    ("Your ethnicity is the reason you act so dumb.", "ethnicity/race"),
    ("Typical behavior from someone of your race.", "ethnicity/race"),
    ("Your community should stay off social media.", "ethnicity/race"),
    ("No wonder you failed, people like you never succeed.", "ethnicity/race"),
    ("Your accent makes you sound uneducated.", "ethnicity/race"),
    ("Every post from your race is annoying.", "ethnicity/race"),
    ("You people are all the same.", "ethnicity/race"),
    ("Why are people from your ethnicity always causing problems?", "ethnicity/race"),
    # Gender/Sexual
    ("Girls can't understand technology properly.", "gender/sexual"),
    ("Boys who cry are weak and pathetic.", "gender/sexual"),
    ("Nobody takes women gamers seriously.", "gender/sexual"),
    ("You act too feminine to be respected.", "gender/sexual"),
    ("Men like you are completely useless.", "gender/sexual"),
    ("People like you shouldn't talk about relationships.", "gender/sexual"),
    ("Your gender identity is just for attention.", "gender/sexual"),
    ("Stop pretending to be something you're not.", "gender/sexual"),
    ("You'll never fit in because of who you are.", "gender/sexual"),
    ("Only idiots would support someone like you.", "gender/sexual"),
    # Religion
    ("Your religion is the reason society has problems.", "religion"),
    ("People from your faith are so brainwashed.", "religion"),
    ("Nobody trusts followers of your religion.", "religion"),
    ("Your beliefs are completely stupid.", "religion"),
    ("Go pray instead of embarrassing yourself online.", "religion"),
    ("Your religion should be banned everywhere.", "religion"),
    ("Only fools follow your religious practices.", "religion"),
    ("Your community is always causing trouble.", "religion"),
    ("People from your religion don't belong here.", "religion"),
    ("I can't believe anyone still follows that religion.", "religion"),
    # Not Cyberbullying
    ("I disagree with your opinion, but I respect your view.", "not_cyberbullying"),
    ("Can you explain your point more clearly?", "not_cyberbullying"),
    ("That game update was disappointing.", "not_cyberbullying"),
    ("I hope you have a good day.", "not_cyberbullying"),
    ("This movie was really enjoyable.", "not_cyberbullying"),
    ("Your presentation was informative and helpful.", "not_cyberbullying"),
    ("I think there's a better way to solve this issue.", "not_cyberbullying"),
    ("Thanks for sharing your thoughts.", "not_cyberbullying"),
    ("I don't personally like this song.", "not_cyberbullying"),
    ("The weather today is really nice.", "not_cyberbullying"),
]

def predict_all(text):
    df = pd.DataFrame({"text": [text], "label": ["not_cyberbullying"]})
    proc = build_preprocessed_dataset(df, for_transformer=False)
    proc_text = proc["processed_text"].iloc[0]
    
    tfidf_pred = tfidf_clf.predict_labels([proc_text])[0]
    
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
    
    distilbert_pred_idx = distilbert_wrapper.predict([text])[0]
    distilbert_pred = distilbert_wrapper.inv_label_map.get(distilbert_pred_idx, distilbert_pred_idx)
    
    distilbert_lora_pred_idx = distilbert_lora_wrapper.predict([text])[0]
    distilbert_lora_pred = distilbert_lora_wrapper.inv_label_map.get(distilbert_lora_pred_idx, distilbert_lora_pred_idx)
    
    return {
        "TF-IDF+SVM": tfidf_pred,
        "LSTM": lstm_pred,
        "DistilBERT": distilbert_pred,
        "DistilBERT+LoRA": distilbert_lora_pred,
    }

# Run tests
print("=" * 80)
print("TESTING NEW EXAMPLES")
print("=" * 80)

correct_counts = {"TF-IDF+SVM": 0, "LSTM": 0, "DistilBERT": 0, "DistilBERT+LoRA": 0}
total = len(TEST_EXAMPLES)
current_category = None

for text, true_label in TEST_EXAMPLES:
    # Print category header
    if true_label != current_category:
        current_category = true_label
        print(f"\n{'='*80}")
        print(f"CATEGORY: {current_category.upper()}")
        print("=" * 80)
    
    preds = predict_all(text)
    status = {}
    for model, pred in preds.items():
        is_correct = pred == true_label
        status[model] = "[OK]" if is_correct else "[FAIL]"
        if is_correct:
            correct_counts[model] += 1
    
    print(f"\nText: {text}")
    print(f"True: {true_label}")
    for model, pred in preds.items():
        print(f"  {model:20s} -> {str(pred):25s} {status[model]}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for model, correct in correct_counts.items():
    pct = (correct / total) * 100
    print(f"{model:20s}: {correct:2d}/{total} correct ({pct:5.1f}%)")

# Save results
results = {
    "test_name": "new_examples_batch",
    "total_examples": total,
    "model_scores": {m: {"correct": c, "total": total, "percentage": round(c/total*100, 1)} for m, c in correct_counts.items()},
    "examples": []
}
for text, true_label in TEST_EXAMPLES:
    preds = predict_all(text)
    results["examples"].append({
        "text": text,
        "true_label": true_label,
        "predictions": preds
    })

with open("reports(New)/new_examples_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[Saved] results -> reports(New)/new_examples_results.json")
