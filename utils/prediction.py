
import re
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences 

# Model configuration used during training
MAX_LENGTH = 200

# Load trained model
model = tf.keras.models.load_model(
    "models/offerguard_bilstm.keras"
)

# Load the tokenizer used during training
with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# Clean input text exactly as during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Predict whether an offer is real or fake
def predict_offer(text):

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("Please provide valid offer text.")

    # Convert text into numerical sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Apply padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Get prediction from trained model
    fake_probability = float(
        model.predict(padded_sequence, verbose=0)[0][0]
    )

    # Classify the offer
    if fake_probability >= 0.5:
        result = "Fake"
        confidence = fake_probability * 100
    else:
        result = "Real"
        confidence = (1 - fake_probability) * 100

    return {
        "prediction": result,
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_probability * 100, 2)
    }
