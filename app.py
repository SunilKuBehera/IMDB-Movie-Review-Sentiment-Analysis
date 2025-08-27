# ...existing code...
import streamlit as st
import time
import numpy as np
from typing import Tuple
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")

# Common model paths to try
_MODEL_PATHS = [
    "notebooks/RNN_model.keras",
    "notebooks/simple_rnn_model.keras",
    "RNN_model.keras",
    "simple_rnn_model.keras",
]

# load IMDB word index (used only for simple token-to-index mapping)
try:
    _IMDB_WORD_INDEX = imdb.get_word_index()
except Exception:
    _IMDB_WORD_INDEX = {}
# reverse index for potential debugging / display
_reverse_word_index = {v: k for k, v in _IMDB_WORD_INDEX.items()}

@st.cache_resource
def load_first_available_model(paths=_MODEL_PATHS):
    """Try to load a Keras model from a list of candidate paths."""
    for p in paths:
        try:
            model = load_model(p)
            return model, p
        except Exception:
            continue
    return None, None

_model, _loaded_model_path = load_first_available_model()

def simple_preprocess(text: str, word_index=_IMDB_WORD_INDEX, maxlen: int = 500):
    """
    Very small tokenizer that maps words to the IMDB indices when available.
    NOTE: For reliable production results, replace this with the exact tokenizer
    used during training (tokenizer + same preprocessing + same maxlen).
    """
    text = text.lower().strip()
    if not text:
        return np.zeros((1, maxlen), dtype=np.int32)
    words = text.split()
    # Keras IMDB word_index maps word -> index (starting at 1). Common practice reserves indices 0-3.
    encoded = []
    for w in words:
        idx = word_index.get(w, 2)  # 2 -> unknown token
        encoded.append(idx)
    padded = sequence.pad_sequences([encoded], maxlen=maxlen, padding="post", truncating="post")
    return padded

def interpret_prediction(preds: np.ndarray) -> Tuple[str, float]:
    """
    Convert raw model output to (label, confidence).
    Supports single-sigmoid output shape (1,1) or 2-class softmax (1,2).
    """
    if preds is None:
        return "Unknown", 0.0
    if preds.ndim == 2 and preds.shape[-1] == 1:
        score = float(preds[0][0])  # sigmoid probability of positive
    elif preds.ndim == 2 and preds.shape[-1] >= 2:
        # treat class 1 as positive
        score = float(preds[0][1])
    else:
        # fallback: use maximum
        score = float(np.max(preds))
    label = "Positive" if score >= 0.5 else "Negative"
    return label, max(0.0, min(1.0, score))

def predict_sentiment(text: str):
    """Preprocess and run inference with loaded model."""
    if _model is None:
        return "ModelMissing", 0.0
    x = simple_preprocess(text, maxlen=getattr(_model, "input_shape", (None, 500))[1])
    preds = _model.predict(x, verbose=0)
    return interpret_prediction(preds)

# ---------- UI (single page, clean) ----------
def app_ui():
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;">
            <h1 style="margin:0;">🎬 IMDB Movie Review Sentiment Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Enter a movie review below and click Analyze. Results show sentiment and confidence.")

    # Layout: two columns (input / controls on left, results + info on right)
    col_input, col_result = st.columns([2, 1])

    with col_input:
        review_text = st.text_area("✍️ Your review", height=100, placeholder="Type or paste a movie review...")
        btn_col1, btn_col2 = st.columns([1,1])
        with btn_col1:
            analyze = st.button("🔍 Analyze", key="analyze")
        with btn_col2:
            clear = st.button("🧹 Clear", key="clear")

        if clear:
            # clear by rerunning with empty text (Streamlit can't directly clear text_area content here)
            st.experimental_rerun()

    # if sample triggered we stored it in session_state; show it
    if " __sample_text" in st.session_state:
        # improbable key; ignore
        pass
    if "__sample_text" in st.session_state and not review_text:
        review_text = st.session_state.get("__sample_text", "")

    # Run analysis and show result on the same page
    if analyze:
        if not review_text or review_text.strip() == "":
            st.warning("Please enter a review to analyze.")
        elif _model is None:
            st.error("No model available to run prediction. Place the trained .keras file in the project and refresh.")
        else:
            # animate progress + run prediction
            with st.spinner("Analyzing review..."):
                progress = st.progress(0)
                for i in range(0, 80, 8):
                    time.sleep(0.03)
                    progress.progress(i + 1)
                label, score = predict_sentiment(review_text)
                for i in range(80, 101, 4):
                    time.sleep(0.02)
                    progress.progress(i)
                progress.empty()

            # Show result card
            emoji = "🌟" if label == "Positive" else "😞" if label == "Negative" else "❓"
            st.markdown(f"### {emoji} {label}")
            st.metric("Confidence", f"{score:.2%}")
            st.progress(score)

            # Extra: show decoded short preview using reverse index (best-effort)
            if _reverse_word_index:
                try:
                    sample_encoded = simple_preprocess(review_text)[0]
                    shown = []
                    for idx in sample_encoded[:30]:
                        if idx == 0:
                            break
                        word = _reverse_word_index.get(int(idx), "?")
                        shown.append(word)
                    st.caption("Token preview : " + " ".join(shown))
                except Exception:
                    pass

            # celebratory animation for strong positive
            if label == "Positive" and score >= 0.85:
                st.balloons()
# ---------- end UI ----------

if __name__ == "__main__":
    app_ui()
