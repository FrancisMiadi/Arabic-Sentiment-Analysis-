import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load your dataset
data = pd.read_csv("12.csv")  


# Simple preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = "".join([char for char in text if char.isalpha() or char.isspace()])
    return text


data["text"] = data["text"].apply(preprocess_text)

# Split data into train and test sets
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text data
vectorizer = TfidfVectorizer(
    max_features=3000
)  # Limit features to speed up computation
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a faster SVM model
svm_model = LinearSVC()  
svm_model.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer
joblib.dump(svm_model, "optimized_svm_model.pkl")
joblib.dump(vectorizer, "svm_optimized_tfidf_vectorizer.pkl")

# Make predictions
y_pred = svm_model.predict(X_test_tfidf)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Save performance metrics to a file
with open("optimized_model_performance.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write(f"Classification Report:\n{class_report}\n")

print("Optimized model and performance files saved successfully!")
