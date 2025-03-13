"""
Osama Zeidan - 1210601
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load the dataset
file_path = "12.csv"
data = pd.read_csv(file_path)

text_column = "text"  # Replace with the actual text column name
label_column = "label"  # Replace with the actual label column name

# Preprocess the dataset
# Check for missing values
data = data.dropna(subset=[text_column, label_column])

# Splitting the data into features and target
X = data[text_column]
y = data[label_column]

# Encode labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder for future inference
label_encoder_filename = "label_encoder.pkl"
joblib.dump(label_encoder, label_encoder_filename)
print(f"Label encoder saved as {label_encoder_filename}")

# Tokenize and pad text data
max_words = 10000  # Maximum number of words in the vocabulary
max_len = 100  # Maximum length of sequences

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_seq, maxlen=max_len, padding="post", truncating="post")

# Save the tokenizer for future inference
tokenizer_filename = "tokenizer.pkl"
joblib.dump(tokenizer, tokenizer_filename)
print(f"Tokenizer saved as {tokenizer_filename}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_encoded, test_size=0.2, random_state=42
)

# Build the neural network model
model = Sequential(
    [
        Embedding(max_words, 128, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(np.unique(y_encoded)), activation="softmax"),  # Output layer
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Add early stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)

# Save the trained model
model_filename = "neural_net_sentiment_model.h5"
model.save(model_filename)
print(f"Neural network model saved as {model_filename}")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save training history if needed
history_data = {
    "accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"],
}
history_filename = "training_history.pkl"
joblib.dump(history_data, history_filename)
print(f"Training history saved as {history_filename}")
