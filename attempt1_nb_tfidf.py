import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess train data
train_essays = pd.read_csv('train_data/train_essays.csv')
train_prompts = pd.read_csv('train_data/train_prompts.csv')
train_data = pd.merge(train_essays, train_prompts, on='prompt_id', how='left')

# Combine relevant fields
train_data['full_text'] = train_data['prompt_name'] + ' ' + train_data['instructions'] + ' ' + train_data['source_text'] + ' ' + train_data['text']

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(train_data['full_text'], train_data['generated'], test_size=0.2, random_state=42)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
                                           
# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Validate the model
# X_valid = X_valid.reset_index(drop=True)
# X_valid.drop('generated', inplace=True)
X_valid_tfidf = tfidf_vectorizer.transform(X_valid)
predictions_valid = model.predict(X_valid_tfidf)


# Evaluate the model on validation data
accuracy_valid = accuracy_score(y_valid, predictions_valid)
print(f'Accuracy on Validation Data: {accuracy_valid}')
print(classification_report(y_valid, predictions_valid, zero_division=1))

# Load and preprocess test data
test_essays = pd.read_csv('test_data/test_essays.csv')
test_data = pd.merge(test_essays, train_prompts, on='prompt_id', how='left')
test_data['full_text'] = test_data['prompt_name'] + ' ' + test_data['instructions'] + ' ' + test_data['source_text'] + ' ' + test_data['text']
test_data['full_text'].fillna('', inplace=True)

# Make predictions on test data
X_test_tfidf = tfidf_vectorizer.transform(test_data['full_text'])
predictions = model.predict(X_test_tfidf)
test_data['generated'] = predictions

# Evaluate the model on test data
accuracy = accuracy_score(test_data['generated'], predictions)
print(f'Accuracy on Test Data: {accuracy}')
print(classification_report(test_data['generated'], predictions))
