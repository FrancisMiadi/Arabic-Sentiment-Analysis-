'''
Osama Zeidan - 1210601
'''
# Import Libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import nltk

# Download NLTK resources
nltk.download("stopwords")

# Define Arabic Stopwords and Stemmer
arabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()


# Preprocessing Function
def preprocess_text(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[ًٌٍَُِْ]", "", text)  # Remove diacritics
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove digits
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in arabic_stopwords]
    return " ".join(words)


# Load Dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data["text"] = data["text"].apply(preprocess_text)
    return data


# Train and Evaluate Model
def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    return model


# Main Execution
def main():
    # Load and preprocess dataset
    dataset_file = "12.csv"
    data = load_dataset(dataset_file)

    # Feature Extraction
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(data["text"])
    y = data["label"]

    # Feature Selection
    selector = SelectKBest(chi2, k=1000)
    X = selector.fit_transform(X, y)

    # Split Dataset
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models to Evaluate
    models = {
        "Linear SVC": LinearSVC(
            C=1.0, class_weight="balanced", multi_class="ovr", random_state=42
        ),
        # "Decision Tree": DecisionTreeClassifier(random_state=42),
        # "Random Forest": RandomForestClassifier(random_state=42, n_estimators=10),
        # "Multinomial NB": MultinomialNB(),
    }

    # Train and Evaluate Each Model
    for model_name, model in models.items():
        train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name)

    # Save Best Model
    best_model = LinearSVC()  # Replace with the model you found best
    best_model.fit(x_train, y_train)
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(selector, "feature_selector.pkl")
    print("\nBest Model Saved Successfully!")

    # Test New Input
    new_text = ["سيئ للغاية ما بفضل استخدمو لانو غبي"]
    new_text_processed = [preprocess_text(text) for text in new_text]
    new_features = vectorizer.transform(new_text_processed)
    new_features_selected = selector.transform(new_features)
    new_prediction = best_model.predict(new_features_selected)
    print("\nNew Input Sentiment Prediction:", new_prediction)


# Run Main Function
if __name__ == "__main__":
    main()
