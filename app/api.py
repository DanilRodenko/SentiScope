from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = FastAPI()

class Review(BaseModel):
    text: str

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.load_state_dict(torch.load('models/bert_model.pt', map_location='cpu'))
model.eval()

@app.post("/predict")
def predict(review: Review):
    tokenized_text = tokenizer(review.text, return_tensors="pt")
    review = model(tokenized_text["input_ids"], tokenized_text["attention_mask"])
    logits = review.logits
    prediction = torch.argmax(logits, dim=-1).item()

    if prediction == 0:
        return {"text": "Negative"}
    else:
        return {"text": "Positive"}