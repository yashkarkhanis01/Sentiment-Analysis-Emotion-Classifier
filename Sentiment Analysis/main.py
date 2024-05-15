# Sentiment Analysis: Emotion Classifier

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("movie_reviews.csv")  # Update with your dataset path

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["Summary"], data["Sentiment"], test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Support Vector Machine (SVM) model
model = SVC(kernel='linear')
model.fit(X_train_vec, y_train)

# Predict sentiment on test data
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define sentiment labels
sentiment_labels = {
    0: "Negative",
    1: "Positive"
}

# Print predicted sentiment
print("Predicted Sentiment:")
for i, sentiment in enumerate(y_pred[:10]):  # Printing first 10 predictions Adjust the range in the loop (y_pred[:10]) if you want to print more or fewer predictions. 
    print(f"Sample {i+1}: {sentiment_labels[sentiment]}")
