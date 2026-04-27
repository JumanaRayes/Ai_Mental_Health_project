import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, GlobalMaxPooling1D,
    Concatenate, Layer, SpatialDropout1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# CUSTOM ATTENTION
class AttentionLayer(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, x):
        score = tf.nn.tanh(tf.matmul(x, self.W))
        weights = tf.nn.softmax(score, axis=1)
        return x * weights


# MAIN CLASS

class EmotionCascadeSystem:

    def __init__(self):

        # Config
        self.MAX_FEATURES = 30000
        self.MAXLEN = 100
        self.EMBED_DIM = 300
        self.BATCH = 256
        self.EPOCHS = 10

        # Paths
        self.data_path = "./Datasets/emotiondata/EmotionDataset.csv"
        self.glove_path = "./gloveEmbed/glove.6B.300d.txt"

        # Objects
        self.df = None
        self.tokenizer = None
        self.embedding_matrix = None
        self.num_words = None

        self.model_gate = None
        self.model_emotion = None
        self.model_noise = None

    
    # CLEAN TEXT
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+|\@\w+|\#|[^\w\s]', '', text)
        return " ".join(text.split())

    # LOAD DATA
    
    def load_dataset(self):

        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)

        self.df.dropna(subset=['text'], inplace=True)
        self.df['text'] = self.df['text'].apply(self.clean_text)
        self.df = self.df[self.df['text'] != ""]

        self.df['is_clear'] = (
            (self.df['positive'] == 1) |
            (self.df['negative'] == 1)
        ).astype(int)

        self.df['is_unclear'] = (
            (self.df['ambiguous'] == 1) |
            (self.df['neutral'] == 1)
        ).astype(int)

        self.df = self.df[self.df['is_clear'] != self.df['is_unclear']]

    # TOKENIZER
    def build_tokenizer(self):

        texts = self.df['text'].values

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.MAX_FEATURES
        )

        self.tokenizer.fit_on_texts(texts)

        X = tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenizer.texts_to_sequences(texts),
            maxlen=self.MAXLEN
        )

        return X

    
    # LOAD GLOVE
    
    def load_glove(self):

        print("Loading GloVe...")
        embeddings_index = {}

        with open(self.glove_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        word_index = self.tokenizer.word_index
        self.num_words = min(self.MAX_FEATURES, len(word_index) + 1)

        self.embedding_matrix = np.zeros(
            (self.num_words, self.EMBED_DIM)
        )

        for word, i in word_index.items():
            if i < self.MAX_FEATURES:
                vec = embeddings_index.get(word)
                if vec is not None:
                    self.embedding_matrix[i] = vec

    # BUILD MODEL
    
    def build_model(self):

        inputs = Input(shape=(self.MAXLEN,))

        x = Embedding(
            self.num_words,
            self.EMBED_DIM,
            weights=[self.embedding_matrix],
            trainable=True
        )(inputs)

        x = SpatialDropout1D(0.3)(x)

        x = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.2)
        )(x)

        x = Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.2)
        )(x)

        x = AttentionLayer()(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)

        x = Concatenate()([avg_pool, max_pool])

        x = BatchNormalization()(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    
    # TRAIN ALL MODELS
   
    def train(self):

        X_all = self.build_tokenizer()
        self.load_glove()

        early_stop = EarlyStopping(
            patience=3,
            restore_best_weights=True
        )

       
        # MODEL 1
       
        print("Training Gatekeeper...")

        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            self.df['is_clear'].values,
            test_size=0.1,
            random_state=42
        )

        self.model_gate = self.build_model()

        self.model_gate.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.EPOCHS,
            batch_size=self.BATCH,
            callbacks=[early_stop]
        )

        
        # MODEL 2
       
        print("Training Emotion Expert...")

        df_clear = self.df[self.df['is_clear'] == 1]

        X_clear = tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenizer.texts_to_sequences(df_clear['text']),
            maxlen=self.MAXLEN
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_clear,
            df_clear['positive'].values,
            test_size=0.1,
            random_state=42
        )

        self.model_emotion = self.build_model()

        self.model_emotion.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.EPOCHS,
            batch_size=self.BATCH,
            callbacks=[early_stop]
        )

        
        # MODEL 3
        print("Training Noise Expert...")

        df_unclear = self.df[self.df['is_unclear'] == 1]

        X_unclear = tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenizer.texts_to_sequences(df_unclear['text']),
            maxlen=self.MAXLEN
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_unclear,
            df_unclear['ambiguous'].values,
            test_size=0.1,
            random_state=42
        )

        self.model_noise = self.build_model()

        self.model_noise.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.EPOCHS,
            batch_size=self.BATCH,
            callbacks=[early_stop]
        )

    # PREDICT
    
    def predict(self, text):

        clean = self.clean_text(text)

        seq = tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenizer.texts_to_sequences([clean]),
            maxlen=self.MAXLEN
        )

        gate = self.model_gate.predict(seq, verbose=0)[0][0]

        if gate >= 0.5:

            pred = self.model_emotion.predict(seq, verbose=0)[0][0]

            return "POSITIVE" if pred >= 0.5 else "NEGATIVE"

        else:

            pred = self.model_noise.predict(seq, verbose=0)[0][0]

            return "AMBIGUOUS" if pred >= 0.5 else "NEUTRAL"

    # LOAD PRETRAINED MODELS
    
    def load_models(self, base_path="emotion_models"):
        print(f"Loading emotion models from {base_path}...")
        try:
            from tensorflow.keras.models import load_model
            
            gate_path = os.path.join(base_path, "gatekeeper.keras")
            emotion_path = os.path.join(base_path, "emotion_expert.keras")
            noise_path = os.path.join(base_path, "noise_expert.keras")
            tokenizer_path = os.path.join(base_path, "tokenizer.pkl")
            
            self.model_gate = load_model(gate_path, custom_objects={'AttentionLayer': AttentionLayer})
            self.model_emotion = load_model(emotion_path, custom_objects={'AttentionLayer': AttentionLayer})
            self.model_noise = load_model(noise_path, custom_objects={'AttentionLayer': AttentionLayer})
            
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
                
            print("All Emotion Models Loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load emotion models. Error: {e}")

    # SAVE
    
def save(self):
    os.makedirs("saved_models", exist_ok=True)

    self.model_gate.save("emotion_models/gatekeeper.keras", save_format="keras")
    self.model_emotion.save("emotion_models/emotion_expert.keras", save_format="keras")
    self.model_noise.save("emotion_models/noise_expert.keras", save_format="keras")

    with open("emotion_models/tokenizer.pkl", "wb") as f:
        pickle.dump(self.tokenizer, f)


# MAIN RUN

if __name__ == "__main__":

    system = EmotionCascadeSystem()

    system.load_dataset()
    system.train()

    print(system.predict("I love this phone"))
    print(system.predict("The package arrived yesterday"))
    print(system.predict("I hate and love this together"))

    system.save()