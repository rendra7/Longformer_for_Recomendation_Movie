import streamlit as st
import joblib
import requests
from transformers import LongformerTokenizer, LongformerModel
from transformers import AutoTokenizer, AutoModel
import torch
import io

# Function to download model from Hugging Face
def download_model(model_url):
    response = requests.get(model_url)
    if response.status_code == 200:
        return joblib.load(io.BytesIO(response.content))
    else:
        st.error("Failed to download model from Hugging Face.")
        return None

# Load models
model_url = "https://huggingface.co/Rendra7/Longformer_recomendation_model/resolve/main/recommendation_model.pkl"
model = download_model(model_url)

if model is None:
    st.error("Model failed to load.")
else:
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    longformer = AutoModel.from_pretrained("allenai/longformer-base-4096")

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
