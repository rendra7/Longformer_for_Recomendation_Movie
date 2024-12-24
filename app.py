import streamlit as st
from transformers import AutoTokenizer, AutoModel
import joblib
from huggingface_hub import hf_hub_download
import pandas as pd
import torch
from io import StringIO
import requests

# Function to load CSV from Hugging Face
def load_csv_from_huggingface(url):
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        return pd.read_csv(csv_data)
    else:
        st.error("Failed to load CSV file from Hugging Face.")
        return None

# Load movie dataset from Hugging Face
csv_url = "https://huggingface.co/Rendra7/Longformer_recomendation_model/resolve/main/movie.csv"
df_nlp = load_csv_from_huggingface(csv_url)

# Load models from Hugging Face Hub
@st.cache_resource  # Cache resource to avoid re-downloading
def load_model():
    try:
        # Download model file from Hugging Face
        model_path = hf_hub_download(repo_id="Rendra7/Longformer_recomendation_model", filename="recommendation_model.pkl")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load tokenizer and Longformer model
@st.cache_resource  # Cache model to save memory and improve performance
def load_longformer():
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = AutoModel.from_pretrained("allenai/longformer-base-4096")
    return tokenizer, model

# Streamlit app
st.title("Movie Recommendation System")
st.write("Enter a movie description to get recommendations:")

# Load model and tokenizer
model = load_model()
tokenizer, longformer = load_longformer()

if model and df_nlp is not None:
    # Input from user
    user_input = st.text_input("Enter movie description:")
    if user_input:
        # Generate embedding
        tokens = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        with torch.no_grad():
            embedding = longformer(**tokens).last_hidden_state.mean(dim=1).numpy()

        # Get recommendations
        distances, indices = model.kneighbors(embedding, n_neighbors=6)

        # Display recommendations
        st.write("Recommendations based on your input:")
        for i in range(1, len(indices.flatten())):  # Exclude the input itself (index 0)
            recommended_movie = df_nlp['Title'][indices.flatten()[i]]
            st.write(f"{i}: {recommended_movie}")
else:
    st.error("Model or movie dataset could not be loaded.")
