# Cyberbullying Detector + Safe Reply Generator

AI-powered detection of online hate speech with intelligent, de-escalatory reply generation directed at the aggressor.

## Features

- **4 Detection Models**: TF-IDF + SVM, LSTM, DistilBERT, DistilBERT + LoRA
- **Multi-class Classification**: Detects ethnicity/race, gender/sexual, religion, and not_cyberbullying
- **Safe Reply Generator**: Fine-tuned FLAN-T5-small generates calm, firm, educational de-escalation responses
- **Streamlit Frontend**: Clean, interactive web UI for real-time detection and reply generation
- **CPU-Only**: Runs entirely on CPU — no NVIDIA GPU required

## Quick Start (For End Users)

### One-Click Setup (Windows)

1. Download **`setup.bat`** from the [latest release](https://github.com/Manas-Kushwaha-99/cyberbullying-detector-safe-reply/releases/latest).
2. Double-click `setup.bat`.
3. Wait for automatic download & installation (~10-15 minutes depending on internet speed).
4. A desktop shortcut **"Cyberbullying Detector"** will be created.
5. Double-click the shortcut to launch the app in your browser.

### Manual Setup

```bash
# Clone the repo
git clone https://github.com/Manas-Kushwaha-99/cyberbullying-detector-safe-reply.git
cd cyberbullying-detector-safe-reply

# Run setup
setup.bat

# Launch app
run.bat
```

The app will open at **http://localhost:8501**.

## Project Structure

```
.
├── app.py                          # Streamlit frontend
├── requirements.txt                # Python dependencies
├── setup.bat                       # One-click auto-setup
├── run.bat                         # Launch app
├── src/
│   ├── config.py                   # Central configuration
│   ├── preprocessing.py            # Text preprocessing
│   └── models/
│       ├── lstm_model.py           # LSTM architecture
│       └── tfidf_svm.py            # TF-IDF + SVM classifier
├── scripts/
│   ├── create_desktop_shortcut.ps1
│   ├── download_models.ps1
│   └── zip_models_for_release.ps1
└── models_new/                     # Trained models (downloaded from release)
```

## Model Performance (V2 Enhanced Dataset)

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| TF-IDF + SVM | 99.25% | 99.19% | ~3s |
| LSTM | 99.11% | 99.02% | ~16 epochs |
| DistilBERT | 99.76% | 99.72% | ~59 min |
| DistilBERT + LoRA | 99.71% | 99.66% | ~50 min |

## Dataset

- **Original**: ~97,990 samples
- **Synthetic V1**: +2,000 samples (explicit patterns)
- **Synthetic V2**: +1,200 samples (implicit patterns)
- **Total**: 103,190 samples
- **Reply Training**: 1,000 de-escalatory reply pairs

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.9+
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: ~1 GB free space for models
- **GPU**: Not required — runs on CPU only

## License

MIT License — free for academic and personal use.

## Citation

If you use this work, please cite:

```bibtex
@software{cyberbullying_detector_2026,
  author = {Kushwaha, Manas},
  title = {Cyberbullying Detector + Safe Reply Generator},
  year = {2026},
  url = {https://github.com/Manas-Kushwaha-99/cyberbullying-detector-safe-reply}
}
```
