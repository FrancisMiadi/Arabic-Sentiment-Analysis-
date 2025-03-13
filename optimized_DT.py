"""
Osama Zeidan - 1210601
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Download NLTK resources if not already downloaded
import nltk

nltk.download("stopwords")

# Step 1: Load the Dataset
file_path = "12.csv"
data = pd.read_csv(file_path)

# Assuming the dataset has columns 'text' and 'label'
text_column = "text"  # Replace with the actual text column name
label_column = "label"  # Replace with the actual label column name

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

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=2000, ngram_range=(1, 2)
)  # Unigrams and bigrams
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Save the vectorizer
vectorizer_filename = "optimized_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_filename)
print(f"Vectorizer saved as {vectorizer_filename}")

# Step 6: Optimize Decision Tree Parameters Using Randomized Search
param_grid = {
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}

random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=10,  # Number of random combinations to try
    cv=5,
    scoring="accuracy",
    n_jobs=-1,  # Use all CPU cores
    random_state=42,
)

# Use a subset of the training data for faster tuning
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train_vect, y_train, test_size=0.8, random_state=42
)
random_search.fit(X_train_sample, y_train_sample)

# Best parameters
print(f"Best parameters: {random_search.best_params_}")
optimized_model = random_search.best_estimator_

# Save the optimized model
model_filename = "optimized_decision_tree_model_fast.pkl"
joblib.dump(optimized_model, model_filename)
print(f"Optimized Decision Tree model saved as {model_filename}")

# Step 7: Evaluate the Optimized Model
y_pred = optimized_model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
