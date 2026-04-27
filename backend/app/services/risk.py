import os
import re
import pickle
from textblob import TextBlob

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Layer
    import tensorflow.keras.backend as K
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Layer = object

if TENSORFLOW_AVAILABLE:
    # Custom Attention Layer required to load the model
    @tf.keras.utils.register_keras_serializable()
    class Attention(Layer):
        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1], 1),
                                     initializer='random_normal',
                                     trainable=True)
            self.b = self.add_weight(shape=(input_shape[1], 1),
                                     initializer='zeros',
                                     trainable=True)

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            output = x * a
            return K.sum(output, axis=1)

        def get_config(self):
            return super().get_config()
else:
    class Attention:
        pass

# Paths relative to backend
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "AIModels",
    "saved_models",
    "risk",
    "bi_gru_attention_risk_model.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "AIModels",
    "saved_models",
    "risk",
    "tokenizer.pkl"
)

MAX_LEN = 120

# Load Model globally
model = None
tokenizer = None
if TENSORFLOW_AVAILABLE:
    try:
        model = load_model(MODEL_PATH, custom_objects={'Attention': Attention})
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load ML models correctly: {e}")
else:
    print("Warning: TensorFlow is not installed. Risk detection will use fallback keywords only.")


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
        'suicid', 'kill myself', 'hopeless',
        'overdose', 'depressed', 'self harm',
        'cant go on', 'help'
    ]

    text_clean = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    model_pred = model.predict(pad, verbose=0)[0][0]

    # smarter keyword boost (NOT override)
    if any(k in text_clean for k in high_risk_keywords):
        if model_pred > 0.3:
            return "RISK", model_pred
        else:
            return "CHECK", model_pred

    return ("RISK" if model_pred > 0.45 else "SAFE", model_pred)


def detect_risk(text):
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