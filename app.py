import streamlit as st
import joblib
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import requests
from io import StringIO

# Function to download CSV from Hugging Face
def load_csv_from_huggingface(url):
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        return pd.read_csv(csv_data)
    else:
        st.error("Failed to load CSV file from Hugging Face.")
        return None

# URL of the CSV file on Hugging Face
csv_url = "https://huggingface.co/Rendra7/Longformer_recomendation_model/resolve/main/movie.csv"

# Load movie dataset
df_nlp = load_csv_from_huggingface(csv_url)

# Load models
model = joblib.load("recommendation_model.pkl")  # Ensure this file exists locally or modify to load from a URL
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
longformer = AutoModel.from_pretrained("allenai/longformer-base-4096")

# Streamlit App
st.title("Movie Recommendation System")
st.write("Enter a movie description to get recommendations:")

if df_nlp is not None:
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
    st.error("Movie dataset could not be loaded.")
