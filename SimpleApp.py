"""
Osama Zeidan - 1210601
"""

from flask import Flask, render_template, request
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Initialize Flask app
app = Flask(__name__)

# Load Models and Preprocessing Artifacts
models = {
    "Decision Tree (Default)": {
        "model": joblib.load("decision_tree_sentiment_model.pkl"),
        "vectorizer": joblib.load("vectorizer.pkl"),
    },
    "Decision Tree (Optimized)": {
        "model": joblib.load("optimized_decision_tree_model_fast.pkl"),
        "vectorizer": joblib.load("optimized_vectorizer.pkl"),
    },
    "Naive Bayes (Default)": {
        "model": joblib.load("naive_bayes_sentiment_model.pkl"),
        "vectorizer": joblib.load("vectorizer.pkl"),
    },
    "Naive Bayes (Optimized)": {
        "model": joblib.load("optimized_naive_bayes_model.pkl"),
        "vectorizer": joblib.load("optimized_vectorizer_nb.pkl"),
    },
    "Neural Network (Default)": {
        "model": load_model("neural_net_sentiment_model.h5"),
        "tokenizer": joblib.load("tokenizer.pkl"),
        "label_encoder": joblib.load("label_encoder.pkl"),
    },
    "Neural Network (Optimized)": {
        "model": load_model("optimized_neural_network_model.h5"),
        "tokenizer": joblib.load("optimized_tokenizer_nn.pkl"),
        "label_encoder": joblib.load("optimized_label_encoder_nn.pkl"),
    },
    "Linear SVM": {
        "model": joblib.load("optimized_svm_model.pkl"),
        "vectorizer": joblib.load("svm_optimized_tfidf_vectorizer.pkl"),
    },
}


# Simple Text Preprocessing
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip().lower()  # Convert to lowercase


# Analyze sentiment based on the selected model
def analyze_sentiment(model_name, input_text):
    model_group = models[model_name]
    processed_text = preprocess_text(input_text)
    if "vectorizer" in model_group:  # For Decision Tree and Naive Bayes
        vectorizer = model_group["vectorizer"]
        model = model_group["model"]
        input_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(input_vectorized)
        return prediction[0], processed_text
    elif "tokenizer" in model_group:  # For Neural Networks
        tokenizer = model_group["tokenizer"]
        model = model_group["model"]
        label_encoder = model_group["label_encoder"]
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding="post")
        prediction = model.predict(padded_sequence).argmax(axis=-1)
        return label_encoder.inverse_transform(prediction)[0], processed_text


# Get the color based on the sentiment
def get_color(sentiment):
    sentiment_colors = {
        "positive": "success",  # Green
        "neutral": "primary",  # Blue
        "negative": "danger",  # Red
    }
    return sentiment_colors.get(sentiment.lower(), "secondary")  # Default gray


# Define routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_name = None
    input_text = None
    color = None
    preprocessed_text = None
    if request.method == "POST":
        model_name = request.form["model"]
        input_text = request.form["text"]
        prediction, preprocessed_text = analyze_sentiment(model_name, input_text)
        color = get_color(prediction)
    return render_template(
        "index.html",
        models=models.keys(),
        prediction=prediction,
        preprocessed_text=preprocessed_text,
        input_text=input_text,
        model_name=model_name,
        color=color,
    )


if __name__ == "__main__":
    app.run(debug=True)
