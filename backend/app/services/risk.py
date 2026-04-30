import os
import pickle
import re

from textblob import TextBlob
from AIModels.risk_detection import AttentionLayer

try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Layer = object


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "AIModels" / "saved_models" / "risk"
# MODEL_PATH = (
#  BASE_DIR
# / "AIModels"
# / "saved_models"
# / "risk"
# / "bi_gru_attention_risk_model.keras"
# )
TOKENIZER_PATH = BASE_DIR / "AIModels" / "saved_models" / "risk" / "tokenizer.pkl"


print("FILE LOCATION:", __file__)
print("DIRNAME:", os.path.dirname(__file__))
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)


MAX_LEN = 120


# Load Model globally
model = None
tokenizer = None
if TENSORFLOW_AVAILABLE:
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False,
        )

        print("Model loaded successfully.")

    except Exception as e:
        print(f"Warning: Could not load ML models correctly: {e}")
else:
    print(
        "Warning: TensorFlow is not installed. Risk detection will use fallback keywords only."
    )


def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def predict_risk(text):
    # This shouldn't be reached if model is None, but adding protection
    if model is None or tokenizer is None:
        return "SAFE", 0.0

    high_risk_keywords = [
        "suicid",
        "kill myself",
        "hopeless",
        "overdose",
        "depressed",
        "self harm",
        "cant go on",
        "help",
    ]

    text_clean = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    model_pred = model.predict(pad, verbose=0)[0][0]

    # smarter keyword boost (NOT override)
    if any(k in text_clean for k in high_risk_keywords):
        if model_pred > 0.3:
            return "RISK", model_pred
        else:
            return "CHECK", model_pred

    return ("RISK" if model_pred > 0.45 else "SAFE", model_pred)


def load_model_once():
    global model, tokenizer

    if model is not None:
        return model

    tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
        safe_mode=False,
    )

    return model


def detect_risk(text):
    model = load_model_once()

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=120, padding="post")

    score = model.predict(pad, verbose=0)[0][0]

    return {"type": "warning" if score > 0.45 else "safe", "score": float(score)}


def _legacydetect_risk(text):
    # Dummy fallback if ML Model failed to load (e.g. tensorflow not installed)
    if model is None or tokenizer is None or not TENSORFLOW_AVAILABLE:
        keywords = ["suicide", "kill myself", "die"]
        for word in keywords:
            if word in text.lower():
                return {"type": "risk", "score": 1.0}
        return {"type": "safe", "score": 0.0}

    label, score = predict_risk(text)

    if label == "RISK":
        return {"type": "risk", "score": float(score)}
    elif label == "CHECK":
        return {"type": "warning", "score": float(score)}
    else:
        return {"type": "safe", "score": float(score)}
