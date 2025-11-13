# 🧠 Hybrid Sentiment Analysis (DistilBERT + BiLSTM)

![CI](https://github.com/<YOUR_USERNAME>/hybrid-sentiment/actions/workflows/ci.yml/badge.svg)
[![Codecov](https://codecov.io/gh/<YOUR_USERNAME>/hybrid-sentiment/branch/main/graph/badge.svg)](https://codecov.io/gh/<YOUR_USERNAME>/hybrid-sentiment)

A hybrid deep-learning pipeline combining **DistilBERT embeddings** and a **BiLSTM network** for robust text sentiment classification.  
Supports full CI/CD, FastAPI, and Streamlit deployment, and Hugging Face Hub integration.

---

## 🚀 Features
- Dual-branch architecture (DistilBERT + BiLSTM)
- TensorFlow 2.14 + Transformers
- CI/CD via GitHub Actions + Codecov
- FastAPI REST API
- Streamlit interactive UI
- Docker / Docker Compose support

---

## 🧩 Quick Start (Local)

```bash
git clone [https://github.com/ezana-fekadu/hybrid-sentiment.git](https://github.com/Ezana-Fekadu/hybrid-sentiment.git)
cd hybrid-sentiment
python -m venv venv
source venv/bin/activate   # (Windows: .\\venv\\Scripts\\activate)
pip install -r requirements.txt   # or use pip install commands below
python hybrid_sentiment_harness.py --epochs 1 --batch_size 8 --limit 1000
