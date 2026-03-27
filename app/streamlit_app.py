import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('models/bert_model.pt', map_location='cpu'))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Sentiment Analysis")
text = st.text_area("Enter your review here")
button = st.button('Analyze')

if button and text:
    tokenized_text = tokenizer(text, return_tensors='pt')
    review = model(tokenized_text['input_ids'], tokenized_text['attention_mask'])
    logits = review.logits
    prediction = logits.argmax(dim=1).item()

    if prediction == 1:
        st.success("Positive")
    else:
        st.error("Negative")
elif text is None:
    st.text("Please enter your review here")
