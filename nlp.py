import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Download the required NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

daigt_dataset = pd.read_csv('daigt.csv')
data, labels = daigt_dataset['text'], daigt_dataset['label']


# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # use stemmer to get the root of the word
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    return [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]

# Preprocess the dataset
preprocessed_data = [" ".join(preprocess_text(text)) for text in data]

# perform tf-idf vectorization
vectorizer = TfidfVectorizer(max_features=5000)
features = vectorizer.fit_transform(preprocessed_data)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Save the model
with open('models/knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

# Predict and evaluate
predictions = knn_model.predict(X_test)
print(classification_report(y_test, predictions))
