"""
Osama Zeidan - 1210601
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Download NLTK resources if not already downloaded
import nltk

nltk.download("stopwords")

# Step 1: Load the Dataset
file_path = "12.csv" 
data = pd.read_csv(file_path)

text_column = "text"  
label_column = "label" 

# Check for missing values
data = data.dropna(subset=[text_column, label_column])

# Step 2: Define Preprocessing Functions

# Remove diacritics
arabic_diacritics = re.compile(
    """ ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """,
    re.VERBOSE,
)


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, "", text)
    return text


# Remove repeating characters
def remove_repeating_char(text):
    return re.sub(r"(.)\1+", r"\1\1", text)  # Keep at most 2 repeats


# Normalize and clean text
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    return text.strip()


# Stopword removal
arabic_stopwords = set(stopwords.words("arabic"))


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in arabic_stopwords])


# Apply stemming
stemmer = ISRIStemmer()


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


# Preprocess the text
def preprocess_text(text):
    text = remove_diacritics(text)
    text = remove_repeating_char(text)
    text = clean_text(text)
    text = remove_stopwords(text)
    text = stem_words(text)
    return text


# Step 3: Preprocess the Dataset
data[text_column] = data[text_column].apply(preprocess_text)

# Step 4: Split the Data
X = data[text_column]
y = data[label_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Tokenize and Pad Sequences
max_words = 10000  # Maximum number of words in the vocabulary
max_len = 100  # Maximum length of sequences

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq, maxlen=max_len, padding="post", truncating="post"
)
X_test_pad = pad_sequences(
    X_test_seq, maxlen=max_len, padding="post", truncating="post"
)

# Save the tokenizer for future use
tokenizer_filename = "optimized_tokenizer_nn.pkl"
joblib.dump(tokenizer, tokenizer_filename)
print(f"Tokenizer saved as {tokenizer_filename}")

# Step 6: Encode Labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# Save the label encoder for future use
label_encoder_filename = "optimized_label_encoder_nn.pkl"
joblib.dump(label_encoder, label_encoder_filename)
print(f"Label encoder saved as {label_encoder_filename}")

# Step 7: Build the Neural Network Model
model = Sequential(
    [
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Step 8: Train the Model
early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    X_train_pad,
    y_train_enc,
    validation_data=(X_test_pad, y_test_enc),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)

# Save the trained model
model_filename = "optimized_neural_network_model.h5"
model.save(model_filename)
print(f"Optimized Neural Network model saved as {model_filename}")

# Step 9: Evaluate the Model
loss, accuracy = model.evaluate(X_test_pad, y_test_enc, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification report and confusion matrix
y_pred = model.predict(X_test_pad).argmax(axis=-1)
print("Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))
