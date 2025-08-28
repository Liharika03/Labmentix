import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# ----------------------------
# Load Model & Tokenizer
# ----------------------------
MODEL_PATH = "toxicity_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100

@st.cache_resource
def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_tokenizer(tokenizer_path=TOKENIZER_PATH):
    with open(tokenizer_path, 'rb') as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

# ----------------------------
# NLP Preprocessing
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LEN, padding='post')

# ----------------------------
# Streamlit Tabs
# ----------------------------
st.set_page_config(page_title="Toxic Comment Detector", layout="centered")
tab1, tab2 = st.tabs(["Single Comment", "Bulk Upload"])

# -----------------------------------------
# Tab 1: Single Comment
# -----------------------------------------
with tab1:
    st.header("Check a Single Comment")
    user_input = st.text_area("Enter your comment:")

    if st.button("Check Toxicity", key="single"):
        if user_input.strip() == "":
            st.warning("Please enter a comment.")
        else:
            cleaned = clean_text(user_input)
            padded = preprocess_text(cleaned)
            score = float(model.predict(padded)[0][0])

            # Classification thresholds
            if score <= 0.2:
                st.success("✅ This comment is NON-TOXIC.")
            elif score <= 0.05:
                st.warning("⚠️ This comment is MILDLY TOXIC.")
            elif score <= 0.6:
                st.error("❗ This comment is TOXIC.")
            else:
                st.error("☠️ This comment is HIGHLY TOXIC.")

            st.write(f"Toxicity Score (0-1): **{score:.3f}**")

# -----------------------------------------
# Tab 2: Bulk CSV Upload
# -----------------------------------------
with tab2:
    st.header("Bulk Comment Predictions (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV with a 'comment_text' column", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'comment_text' not in df.columns:
            st.error("CSV must contain a column named 'comment_text'.")
        else:
            st.write("Processing...")
            df['cleaned'] = df['comment_text'].apply(clean_text)
            sequences = tokenizer.texts_to_sequences(df['cleaned'])
            padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
            predictions = model.predict(padded).flatten()

            # Map scores to labels
            def map_label(score):
                if score <= 0.20:
                    return "NON-TOXIC"
                elif score <= 0.50:
                    return "MILDLY TOXIC"
                elif score <= 0.80:
                    return "TOXIC"
                else:
                    return "HIGHLY TOXIC"

            df['toxicity_score'] = predictions
            df['prediction'] = [map_label(s) for s in predictions]

            st.write("### Results")
            st.dataframe(df[['comment_text', 'toxicity_score', 'prediction']])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "toxicity_results.csv", "text/csv")



