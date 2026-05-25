"""
Cyberbullying Detector + Safe Reply Generator
Streamlit Frontend
"""
import os
import sys
import json
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)
from peft import PeftModel

from src.config import (
    DEVICE, MODELS_DIR, LSTM_MAX_LEN, LABELS, NUM_CLASSES,
    LSTM_EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_BIDIRECTIONAL,
    BERT_MAX_LEN, BERT_MODEL_NAME
)
from src.models.lstm_model import CyberbullyingLSTM
from src.models.tfidf_svm import TfidfSvmClassifier
from src.preprocessing import build_preprocessed_dataset

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyberbullying Detector + Safe Reply",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ── Model Loading (Cached) ─────────────────────────────────────────

@st.cache_resource(show_spinner=True)
def load_detection_models():
    """Load all 4 detection models."""
    models = {}
    
    # 1. TF-IDF + SVM
    try:
        tfidf_path = os.path.join(MODELS_DIR, "tfidf_svm.pkl")
        models["tfidf_svm"] = TfidfSvmClassifier.load(tfidf_path)
    except Exception as e:
        st.error(f"Failed to load TF-IDF+SVM: {e}")
    
    # 2. LSTM
    try:
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
        lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), weights_only=True, map_location='cpu'))
        lstm_model.eval()
        models["lstm"] = (lstm_model, lstm_vocab)
    except Exception as e:
        st.error(f"Failed to load LSTM: {e}")
    
    # 3. DistilBERT
    try:
        distilbert_path = os.path.join(MODELS_DIR, "distilbert")
        distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_path)
        distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path).to(DEVICE)
        distilbert_model.eval()
        models["distilbert"] = (distilbert_model, distilbert_tokenizer)
    except Exception as e:
        st.error(f"Failed to load DistilBERT: {e}")
    
    # 4. DistilBERT + LoRA
    try:
        distilbert_lora_path = os.path.join(MODELS_DIR, "distilbert_lora")
        distilbert_lora_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_lora_path)
        base_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
        distilbert_lora_model = PeftModel.from_pretrained(base_model, distilbert_lora_path).to(DEVICE)
        distilbert_lora_model.eval()
        models["distilbert_lora"] = (distilbert_lora_model, distilbert_lora_tokenizer)
    except Exception as e:
        st.error(f"Failed to load DistilBERT+LoRA: {e}")
    
    return models

@st.cache_resource(show_spinner=True)
def load_reply_model():
    """Load FLAN-T5-small reply generator."""
    try:
        model_path = os.path.join(MODELS_DIR, "flan_t5_small_reply")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load reply generator: {e}")
        return None, None

# ── Prediction Functions ───────────────────────────────────────────

def predict_tfidf(text, model):
    df = pd.DataFrame({"text": [text], "label": ["not_cyberbullying"]})
    proc = build_preprocessed_dataset(df, for_transformer=False)
    proc_text = proc["processed_text"].iloc[0]
    pred = model.predict_labels([proc_text])[0]
    # Get decision function scores for confidence
    try:
        scores = model.pipeline.decision_function([proc_text])[0]
        # Convert to softmax-like probabilities
        exp_scores = torch.tensor(scores).exp()
        probs = (exp_scores / exp_scores.sum()).numpy()
        return pred, {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    except:
        return pred, {label: 0.25 for label in LABELS}

def predict_lstm(text, model, vocab):
    df = pd.DataFrame({"text": [text], "label": ["not_cyberbullying"]})
    proc = build_preprocessed_dataset(df, for_transformer=False)
    proc_text = proc["processed_text"].iloc[0]
    
    tokens = proc_text.split()
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(indices) < LSTM_MAX_LEN:
        indices += [vocab["<PAD>"]] * (LSTM_MAX_LEN - len(indices))
    else:
        indices = indices[:LSTM_MAX_LEN]
    
    with torch.no_grad():
        lstm_input = torch.tensor([indices], dtype=torch.long).to(DEVICE)
        outputs = model(lstm_input)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    pred = LABELS[torch.argmax(outputs, 1).item()]
    return pred, {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

def predict_distilbert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=BERT_MAX_LEN)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    pred = LABELS[int(torch.argmax(outputs.logits, 1).item())]
    return pred, {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

def predict_distilbert_lora(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=BERT_MAX_LEN)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    pred = LABELS[int(torch.argmax(outputs.logits, 1).item())]
    return pred, {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

def generate_reply(text, model, tokenizer):
    instruction = "respond to this hate speech with a calm, respectful de-escalation: "
    input_text = instruction + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ── UI Helpers ─────────────────────────────────────────────────────

def get_label_emoji(label):
    emoji_map = {
        "not_cyberbullying": "✅ Not Cyberbullying",
        "ethnicity/race": "⚠️ Ethnicity / Race",
        "gender/sexual": "⚠️ Gender / Sexual",
        "religion": "⚠️ Religion"
    }
    return emoji_map.get(label, label)

# ── Main App ───────────────────────────────────────────────────────

def main():
    st.markdown('<div class="main-header">🛡️ Cyberbullying Detector + Safe Reply Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered detection of online hate speech with intelligent de-escalation responses</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            "This app uses 4 machine learning models to detect cyberbullying: "
            "**TF-IDF + SVM** (fast, lightweight), "
            "**LSTM** (sequential text understanding), "
            "**DistilBERT** (deep contextual understanding), and "
            "**DistilBERT + LoRA** (efficient fine-tuning). "
            "The **Safe Reply Generator** uses FLAN-T5-small to generate "
            "calm, respectful de-escalation responses directed at the person "
            "saying the hate speech."
        )
        if str(DEVICE) == "cpu":
            st.warning("Running on CPU. Detection and reply generation will be slower.")
    
    # Tabs
    tab1, tab2 = st.tabs(["🔍 Detection", "💬 Reply Generator"])
    
    # ── Tab 1: Detection ──────────────────────────────────────────
    with tab1:
        st.header("Cyberbullying Detection")
        st.markdown("Enter text to analyze. All 4 models will predict the category.")
        
        text_input = st.text_area(
            "Text to analyze:",
            height=100,
            placeholder="Type or paste text here..."
        )
        
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Loading models and analyzing..."):
                    models = load_detection_models()
                
                if not models:
                    st.error("Failed to load models. Please check the models directory.")
                    return
                
                results = []
                
                # TF-IDF + SVM
                if "tfidf_svm" in models:
                    pred, probs = predict_tfidf(text_input, models["tfidf_svm"])
                    results.append(("TF-IDF + SVM", pred, probs))
                
                # LSTM
                if "lstm" in models:
                    pred, probs = predict_lstm(text_input, models["lstm"][0], models["lstm"][1])
                    results.append(("LSTM", pred, probs))
                
                # DistilBERT
                if "distilbert" in models:
                    pred, probs = predict_distilbert(text_input, models["distilbert"][0], models["distilbert"][1])
                    results.append(("DistilBERT", pred, probs))
                
                # DistilBERT + LoRA
                if "distilbert_lora" in models:
                    pred, probs = predict_distilbert_lora(text_input, models["distilbert_lora"][0], models["distilbert_lora"][1])
                    results.append(("DistilBERT + LoRA", pred, probs))
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Individual model results
                cols = st.columns(len(results))
                for idx, (model_name, pred, probs) in enumerate(results):
                    with cols[idx]:
                        st.markdown(f"**{model_name}**")
                        st.markdown(f"### {pred}")

                        # Show confidence bars
                        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
                            st.progress(min(prob, 1.0), text=f"{label}: {prob*100:.1f}%")
                
                # Majority vote
                preds = [r[1] for r in results]
                from collections import Counter
                majority = Counter(preds).most_common(1)[0][0]
                
                if majority != "not_cyberbullying":
                    st.info("Cyberbullying detected! Go to the **Reply Generator** tab to generate a de-escalation response.")

        # Detection-only examples
        st.markdown("---")
        st.header("Examples")
        st.caption("Copy-paste any example into the text box above to try them out.")

        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.subheader("Ethnicity / Race")
            st.code("Go back to where you came from, nobody wants you here.", language="text")
            st.code("People from your race are always ruining this country.", language="text")
            st.code("Go back to your country, you don't belong here.", language="text")
            st.code("Your culture is completely inferior to ours.", language="text")
            st.code("Why do people like you always cause trouble here?", language="text")

            st.subheader("Gender / Sexual")
            st.code("Women are way too emotional to ever hold a leadership position.", language="text")
            st.code("Girls can't handle technical work.", language="text")
            st.code("You only got promoted because of diversity quotas.", language="text")
            st.code("Women belong in the kitchen, not in the boardroom.", language="text")
            st.code("Men are naturally superior at logical reasoning.", language="text")

        with col_ex2:
            st.subheader("Religion")
            st.code("Your religion is a violent cult that needs to be banned.", language="text")
            st.code("Your religion should be banned everywhere.", language="text")
            st.code("Go pray instead of embarrassing yourself.", language="text")
            st.code("All terrorists follow your religion.", language="text")
            st.code("That belief system is completely backwards.", language="text")

            st.subheader("Not Cyberbullying")
            st.code("I disagree with your opinion, but I respect your view.", language="text")
            st.code("Let's discuss this calmly and find common ground.", language="text")
            st.code("I appreciate your perspective, even though I see it differently.", language="text")
            st.code("Can you explain your reasoning? I want to understand better.", language="text")
            st.code("Thanks for sharing your thoughts on this topic.", language="text")

    # -- Tab 2: Reply Generator -------------------------------------
    with tab2:
        st.header("Safe Reply Generator")
        st.markdown("Enter cyberbullying text to generate a calm, respectful de-escalation response **directed at the aggressor**.")

        reply_input = st.text_area(
            "Cyberbullying text:",
            height=100,
            placeholder="Enter the hate speech text here...",
            key="reply_input"
        )

        if st.button("Generate De-escalation Reply", type="primary", use_container_width=True):
            if not reply_input.strip():
                st.warning("Please enter some text to generate a reply for.")
            else:
                with st.spinner("Generating de-escalation response..."):
                    reply_model, reply_tokenizer = load_reply_model()

                if reply_model is None:
                    st.error("Failed to load reply generator model.")
                    return

                reply = generate_reply(reply_input, reply_model, reply_tokenizer)

                st.markdown("---")
                st.subheader("De-escalation Reply (to aggressor)")

                # Copyable reply block
                st.code(reply, language="text")
                st.caption("Copy the reply above to use it in your response.")

                # Contextual tips
                st.markdown("---")
                st.subheader("Tips for Using This Reply")
                st.markdown("""
                - **Stay calm**: The reply is designed to be non-confrontational
                - **Be firm**: It clearly states that the behavior is unacceptable
                - **Educate**: It gently corrects false assumptions
                - **Don't escalate**: Avoid getting into a back-and-forth argument
                - **Report if needed**: If the harassment continues, use platform reporting tools
                """)

        # Reply Generator examples (no "Not Cyberbullying")
        st.markdown("---")
        st.header("Examples")
        st.caption("Copy-paste any example into the text box above to try them out.")

        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.subheader("Ethnicity / Race")
            st.code("Go back to where you came from, nobody wants you here.", language="text")
            st.code("People from your race are always ruining this country.", language="text")
            st.code("Go back to your country, you don't belong here.", language="text")
            st.code("Your culture is completely inferior to ours.", language="text")
            st.code("Why do people like you always cause trouble here?", language="text")

            st.subheader("Gender / Sexual")
            st.code("Women are way too emotional to ever hold a leadership position.", language="text")
            st.code("Girls can't handle technical work.", language="text")
            st.code("You only got promoted because of diversity quotas.", language="text")
            st.code("Women belong in the kitchen, not in the boardroom.", language="text")
            st.code("Men are naturally superior at logical reasoning.", language="text")

        with col_ex2:
            st.subheader("Religion")
            st.code("Your religion is a violent cult that needs to be banned.", language="text")
            st.code("Your religion should be banned everywhere.", language="text")
            st.code("Go pray instead of embarrassing yourself.", language="text")
            st.code("All terrorists follow your religion.", language="text")
            st.code("That belief system is completely backwards.", language="text")

if __name__ == "__main__":
    main()
