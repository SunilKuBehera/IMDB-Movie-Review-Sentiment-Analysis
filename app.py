import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

# load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# try multiple common model locations
_MODEL_PATHS = [
    "notebooks/RNN_model.keras",
    "notebooks/simple_rnn_model.keras",
    "RNN_model.keras",
    "simple_rnn_model.keras",
]

_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    for p in _MODEL_PATHS:
        try:
            _model = load_model(p)
            return _model
        except Exception:
            continue
    return None

# Load model (cached)
_model = _load_model()

# Function to decode an encoded review
def decode_review(encoded_review, reverse_word_index):
    return " ".join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input for model prediction
def preprocess_text(text, word_index, maxlen=500):
    # Lowercase and split the text
    words = text.lower().split()
    # Convert words to integer indices
    encoded = [word_index.get(word, 2) for word in words]  # 2 = unknown token
    # Pad the sequence to the desired length
    padded = sequence.pad_sequences([encoded], maxlen=maxlen, padding='post', truncating='post')
    return padded

# Prediction function
def predict_sentiment(review):
    if _model is None:
        # fallback: return neutral/placeholder if model not found
        return "Unknown", 0.0
    preprocess_input = preprocess_text(review, word_index)
    prediction = _model.predict(preprocess_input)
    # handle outputs that may be shape (1,1) or (1,2)
    score = float(prediction[0][0]) if prediction.shape[-1] == 1 else float(prediction[0].max())
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score

def main():
    st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")
    st.title("🎬 IMDB Movie Review Sentiment Analysis 🎬")
    st.markdown("Welcome to the IMDB Movie Review Sentiment Analysis app! 🌟")
    
    st.sidebar.header("User Input")
    user_input = st.sidebar.text_area("Enter your movie review:", height=200)

    if st.sidebar.button("Analyze"):
        if not user_input:
            st.warning("Please enter a review to analyze.")
        else:
            if _model is None:
                st.error("Model file not found. Expected one of: " + ", ".join(_MODEL_PATHS))
                st.info("Place your .keras model in the project or update the path.")
            else:
                with st.spinner("Analyzing..."):
                    sentiment, score = predict_sentiment(user_input)
                emoji = "🌟" if sentiment == "Positive" else "😔" if sentiment == "Negative" else "❓"
                st.success(f"Sentiment: {sentiment} {emoji}")
                st.write(f"Confidence Score: {score:.2f}")
                st.progress(min(max(score, 0.0), 1.0))

    st.markdown("### How it works:")
    st.write("This application uses a trained sentiment analysis model to predict the sentiment of your movie review.")
    st.write("Enter your review in the sidebar and click 'Analyze' to see the results.")

if __name__ == "__main__":
    main()