"""
Osama Zeidan - 1210601
"""

import joblib

# Load the saved model and vectorizer
model_filename = "optimized_naive_bayes_model.pkl"
vectorizer_filename = "optimized_vectorizer_nb.pkl"

model = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)


def predict_sentiment(text):
    """
    Predict the sentiment of a given text using the loaded model and vectorizer.

    Args:
        text (str): The input text to classify.

    Returns:
        str: The predicted sentiment label.
    """
    # Preprocess the input text (vectorization)
    text_vect = vectorizer.transform([text])

    # Predict the sentiment
    prediction = model.predict(text_vect)

    # Return the predicted label
    return prediction[0]


# Example usage
new_text = "" 
predicted_sentiment = predict_sentiment(new_text)
print(f"The predicted sentiment is: {predicted_sentiment}")
