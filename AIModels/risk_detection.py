import os
import re
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    recall_score
)

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import (
    Layer,
    Input,
    Embedding,
    Bidirectional,
    GRU,
    Dense,
    Dropout
)

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# ATTENTION LAYER
# =========================

class AttentionLayer(Layer):

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


# =========================
# RISK SYSTEM
# =========================

class RiskDetectionSystem:

    def __init__(self):

        self.path1 = "./Datasets/riskData/Reddit Dataset_cleaned.csv"
        self.path2 = "./Datasets/riskData/mental_health_cleaned_final.csv"

        self.save_dir = "saved_models/risk"

        self.model_path = os.path.join(
            self.save_dir,
            "bi_gru_attention_risk_model.keras"
        )

        self.tokenizer_path = os.path.join(
            self.save_dir,
            "tokenizer.pkl"
        )

        self.MAX_VOCAB = 20000
        self.MAX_LEN = 120
        self.BATCH = 32
        self.EPOCHS = 10
        self.THRESHOLD = 0.45

        self.df = None
        self.tokenizer = None
        self.model = None

        self.X_train_pad = None
        self.X_val_pad = None
        self.y_train = None
        self.y_val = None

    # =========================
    # CLEAN TEXT
    # =========================
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return text.strip()

    # =========================
    # LOAD DATA
    # =========================
    def load_data(self):

        df1 = pd.read_csv(self.path1)
        df2 = pd.read_csv(self.path2)

        df1["text"] = df1["title"].fillna("") + " " + df1["selftext"].fillna("")
        df1["label_raw"] = df1["Label"].astype(str).str.lower().str.strip()

        risk_categories = [
            "drug and alcohol",
            "personality",
            "trauma and stress",
            "early life"
        ]

        def map_label(x):
            if x == "0":
                return 0
            elif x in risk_categories:
                return 1
            return 0

        df1["label"] = df1["label_raw"].apply(map_label)
        df1 = df1[["text", "label"]]

        df2["label"] = (
            (df2["has_suicidal_keyword"] == 1) |
            (df2["has_help_keyword"] == 1)
        ).astype(int)

        df2 = df2[["text", "label"]]

        self.df = pd.concat([df1, df2], ignore_index=True)
        self.df["text"] = self.df["text"].apply(self.clean_text)

        print("Datasets loaded.")

    # =========================
    # BALANCE DATA
    # =========================
    def balance_data(self):

        df_major = self.df[self.df["label"] == 0]
        df_minor = self.df[self.df["label"] == 1]

        df_major = df_major.sample(
            n=len(df_minor) * 2,
            random_state=42
        )

        self.df = pd.concat([df_major, df_minor])

        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates(subset=["text"])
        self.df = self.df[self.df["text"].str.strip() != ""]

        print("Data balanced.")

    # =========================
    # PREPARE DATA (FIXED)
    # =========================
    def prepare_data(self):

        X_train, X_val, y_train, y_val = train_test_split(
            self.df["text"],
            self.df["label"],
            test_size=0.2,
            stratify=self.df["label"],
            random_state=42
        )

        self.tokenizer = Tokenizer(
            num_words=self.MAX_VOCAB,
            oov_token="<OOV>"
        )

        self.tokenizer.fit_on_texts(X_train)

        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)

        self.X_train_pad = pad_sequences(
            X_train_seq,
            maxlen=self.MAX_LEN,
            padding="post"
        )

        self.X_val_pad = pad_sequences(
            X_val_seq,
            maxlen=self.MAX_LEN,
            padding="post"
        )

        # 🔥 FIX: convert to numpy (THIS FIXES YOUR ERROR)
        self.y_train = np.array(y_train).astype(int)
        self.y_val = np.array(y_val).astype(int)

        print("Data prepared.")

    # =========================
    # BUILD MODEL
    # =========================
    def build_model(self):

        inputs = Input(shape=(self.MAX_LEN,))

        x = Embedding(self.MAX_VOCAB, 128)(inputs)

        x = Bidirectional(GRU(32, return_sequences=True))(x)
        x = Dropout(0.4)(x)

        x = AttentionLayer()(x)

        x = Dense(16, activation="relu")(x)
        x = Dropout(0.3)(x)

        outputs = Dense(1, activation="sigmoid")(x)

        self.model = Model(inputs, outputs)

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("Model built.")

    # =========================
    # TRAIN (FIXED)
    # =========================
    def train(self):

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=self.y_train
        )

        class_weights = {0: class_weights[0], 1: class_weights[1]}

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )

        self.model.fit(
            self.X_train_pad,
            self.y_train,
            validation_data=(self.X_val_pad, self.y_val),
            epochs=self.EPOCHS,
            batch_size=self.BATCH,
            class_weight=class_weights,
            callbacks=[early_stop]
        )

    # =========================
    # EVALUATE
    # =========================
    def evaluate(self):

        probs = self.model.predict(self.X_val_pad)
        preds = (probs > self.THRESHOLD).astype(int)

        print(classification_report(self.y_val, preds))
        print("F1:", f1_score(self.y_val, preds))
        print("ROC-AUC:", roc_auc_score(self.y_val, probs))
        print("Recall:", recall_score(self.y_val, preds))

    # =========================
    # SAVE
    # =========================
    def save(self):

        os.makedirs(self.save_dir, exist_ok=True)

        self.model.save(self.model_path)

        with open(self.tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        print("Model + tokenizer saved successfully.")

    # =========================
    # LOAD
    # =========================
    def load(self):

        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"AttentionLayer": AttentionLayer}
        )

        print("Model loaded successfully.")

    # =========================
    # PREDICT
    # =========================
    def predict(self, text):

        keywords = [
            "suicid", "kill myself", "hopeless",
            "overdose", "depressed", "self harm",
            "cant go on", "help"
        ]

        text = self.clean_text(text)

        seq = self.tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=self.MAX_LEN, padding="post")

        score = self.model.predict(pad, verbose=0)[0][0]

        if any(k in text for k in keywords) and score > 0.30:
            return {"type": "warning", "score": float(score)}

        if score > self.THRESHOLD:
            return {"type": "risk", "score": float(score)}

        return {"type": "safe", "score": float(score)}


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    system = RiskDetectionSystem()

    system.load_data()
    system.balance_data()
    system.prepare_data()

    system.build_model()
    system.train()

    system.evaluate()
    system.save()

    print(system.predict("I feel hopeless and can't go on"))