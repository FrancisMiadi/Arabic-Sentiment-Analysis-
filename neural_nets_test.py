import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model, tokenizer, and label encoder
model_filename = "neural_net_sentiment_model.h5"
tokenizer_filename = "tokenizer.pkl"
label_encoder_filename = "label_encoder.pkl"

model = load_model(model_filename)
tokenizer = joblib.load(tokenizer_filename)
label_encoder = joblib.load(label_encoder_filename)


# Function to predict sentiment
def predict_sentiment(text):
    """
    Predict the sentiment of the given text using the trained neural network model.

    Args:
        text (str): The input text.

    Returns:
        str: Predicted sentiment label.
    """
    # Preprocess the input text
    max_len = 100  # Must match the max_len used during training
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(
        text_seq, maxlen=max_len, padding="post", truncating="post"
    )

    # Predict sentiment
    prediction = model.predict(text_padded)
    predicted_label_index = prediction.argmax(axis=-1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

    return predicted_label


# Example usage
test_text = (
    "بسيطة عادي هو سيء بس ممتاز مش مشكلة كبيرة بسلك منيح" 
)
predicted_sentiment = predict_sentiment(test_text)
print(f"The predicted sentiment for the input text is: {predicted_sentiment}")
