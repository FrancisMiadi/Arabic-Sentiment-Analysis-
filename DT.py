'''
Osama Zeidan - 1210601
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_vect, y_train)

# Save the trained model
model_filename = "decision_tree_sentiment_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Evaluate the model
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the vectorizer for future use
vectorizer_filename = "vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_filename)
print(f"Vectorizer saved as {vectorizer_filename}")
