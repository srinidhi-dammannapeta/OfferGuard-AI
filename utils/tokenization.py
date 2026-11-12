
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load training and testing datasets
train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

# Separate text
X_train = train_data["text"].astype(str)
X_test = test_data["text"].astype(str)

# Tokenizer settings
MAX_WORDS = 20000
MAX_LENGTH = 200

# Create tokenizer
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

# Learn vocabulary from training data only
tokenizer.fit_on_texts(X_train)

# Convert text into numerical sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Apply padding
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# Save tokenizer
with open("models/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Save numerical sequences
import numpy as np

np.save("data/X_train.npy", X_train_pad)
np.save("data/X_test.npy", X_test_pad)

np.save("data/y_train.npy", train_data["label"].values)
np.save("data/y_test.npy", test_data["label"].values)

# Display results
print("========== TOKENIZATION COMPLETED ==========")
print("Vocabulary size:", len(tokenizer.word_index))
print("Training shape:", X_train_pad.shape)
print("Testing shape:", X_test_pad.shape)
print("Maximum sequence length:", MAX_LENGTH)
print("Tokenizer saved successfully!")
