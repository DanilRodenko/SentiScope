# SentiScope 🎬

Sentiment analysis of movie reviews using PyTorch. Two-stage approach: LSTM baseline → BERT fine-tuning.

## Results

| Model | Test Accuracy | Training Time |
|-------|--------------|---------------|
| LSTM (baseline) | 81.87% | ~17 min |
| DistilBERT (fine-tuned) | 92.10% | ~90 min |

## Project Structure
```
SentiScope/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_lstm_training.ipynb
│   ├── 03_bert_finetuning.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── models.py
│   └── preprocessing.py
├── app/
│   ├── streamlit_app.py
│   └── api.py
├── models/
└── requirements.txt
```

## Tech Stack

- PyTorch + MPS (Apple M1)
- HuggingFace Transformers
- DistilBERT (distilbert-base-uncased)
- Streamlit
- FastAPI
- pandas, matplotlib

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Streamlit app
```bash
streamlit run app/streamlit_app.py
```

### FastAPI
```bash
uvicorn app.api:app --reload --port 9000
```

## Dataset

IMDb Movie Reviews — 50k reviews (25k train / 25k test), balanced classes.
Source: HuggingFace Datasets

## Approach

**Stage 1 — LSTM baseline**: Built from scratch with custom tokenizer and vocabulary (~57k tokens). Trained 10 epochs on Apple M1 MPS.

**Stage 2 — DistilBERT fine-tuning**: Fine-tuned pretrained DistilBERT on IMDb. Trained 3 epochs on Google Colab T4 GPU.