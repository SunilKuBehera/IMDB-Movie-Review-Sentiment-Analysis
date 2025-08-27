# IMDB Movie Review Sentiment Analysis |

Lightweight Streamlit web app that predicts the sentiment (positive / negative) of IMDB movie reviews using a trained Keras RNN model.

## Overview
This repository contains:
- A Streamlit UI (`app.py`) that accepts a review string and shows the predicted sentiment and confidence score.
- A prediction entrypoint (`sentiment_model.py`) used by the app to run inference.
- Notebooks used during development and the trained Keras model (`notebooks/simple_rnn_model.keras`).

The current app workflow:
1. User enters a review in the Streamlit sidebar.
2. `app.py` calls `predict_sentiment(text)` from `sentiment_model.py`.
3. The function returns `(sentiment_label, confidence_score)` and the UI shows results with emoji and progress feedback.

## Required packages
Add these to `requirements.txt` (adjust versions as needed):
- streamlit
- tensorflow
- numpy
- pandas
- plotly

## File / Folder Layout
- app.py — Streamlit UI (main application).
- sentiment_model.py — Inference wrapper (load model, preprocess, predict).
- README.md — This document.
- requirements.txt — Python dependencies.
- notebooks/
  - embedding.ipynb — embedding experiments.
  - prediction.ipynb — inference tests.
  - simple_rnn.ipynb — model training notebook.
  - simple_rnn_model.keras — trained Keras model (loadable by Keras).

## Usage notes
- Ensure text preprocessing (tokenizer, padding, lowercasing, cleaning) in `sentiment_model.py` matches what was used during training.
- If your model outputs probabilities for two classes, interpret accordingly (e.g., prob_positive = preds[0][1]).
- Keep large models (the `.keras` file) in the `notebooks/` folder or a `models/` directory and update `_MODEL_PATH` accordingly.

## Development tips
- Work in a virtualenv.
- Use the notebooks to retrain or inspect embeddings/tokenizers.
- Add unit tests around `predict_sentiment()` to ensure consistent preprocessing and predictable outputs.
- For production, consider serving the model with a small REST endpoint (FastAPI) and keep the Streamlit app purely as UI.

## License
MIT License — see LICENSE