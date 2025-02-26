import os
import pickle

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('punkt')

# Load dataset
data = pd.read_csv('IMDB Dataset.csv')
data['sentiment'] = data['sentiment'].str.lower()  # Ensure consistency

# For simplicity, keep only positive/negative (add neutral later if needed)
X = data['review']  # Column name in IMDb dataset
y = data['sentiment']

# Vectorize text
vectorizer = CountVectorizer(stop_words='english', max_features=5000)  # Limit features for speed
X_vectorized = vectorizer.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)  # Increase iterations for convergence
model.fit(X_train, y_train)

# Evaluate (optional)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Save model and vectorizer
model_path = 'sentiment_model.pkl'
vectorizer_path = 'vectorizer.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {os.path.abspath(model_path)}")

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"Vectorizer saved to {os.path.abspath(vectorizer_path)}")

# Verify files
if os.path.getsize(model_path) > 0 and os.path.getsize(vectorizer_path) > 0:
    print("Files saved successfully and are non-empty!")
else:
    print("Error: One or both files are empty.")