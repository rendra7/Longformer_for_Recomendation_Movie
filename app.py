import streamlit as st
import joblib
from transformers import LongformerTokenizer, LongformerModel
import torch

# Load models
model = joblib.load("recommendation_model.pkl")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
longformer = LongformerModel.from_pretrained("allenai/longformer-base-4096")

# Input from user
user_input = st.text_input("Enter movie description:")
if user_input:
    # Generate embedding
    tokens = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        embedding = longformer(**tokens).last_hidden_state.mean(dim=1).numpy()
    
    # Get recommendations
    distances, indices = model.kneighbors(embedding, n_neighbors=5)
    st.write("Recommended Movies:")
    for idx in indices[0]:
        st.write(f"Movie {idx}")  # Replace with actual movie names from your dataset
