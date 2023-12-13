import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and preprocess train data
train_essays = pd.read_csv('train_data/train_essays.csv')
train_prompts = pd.read_csv('train_data/train_prompts.csv')
train_data = pd.merge(train_essays, train_prompts, on='prompt_id', how='left')


# Combine relevant fields
train_data['full_text'] = train_data['prompt_name'] + ' ' + train_data['instructions'] + ' ' + train_data['source_text'] + ' ' + train_data['text']
train_data['text_length'] = train_data['full_text'].apply(len)

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(train_data['full_text'], train_data['generated'], test_size=0.2, random_state=42)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Define the parameter grid for hyperparameter tuning
param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

# Create individual classifiers
nb_classifier = MultinomialNB()
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a VotingClassifier
voting_classifier = VotingClassifier(estimators=[('nb', nb_classifier), ('rf', rf_classifier)], voting='soft')

# Use StratifiedKFold for cross-validation
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Perform grid search with stratified cross-validation
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=stratified_kfold, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Get the best hyperparameters
best_alpha = grid_search.best_params_['alpha']

# Train the model with the best hyperparameters
nb_classifier = MultinomialNB(alpha=best_alpha)
nb_classifier.fit(X_train_tfidf, y_train)

# Validate the model
X_valid_tfidf = tfidf_vectorizer.transform(X_valid)
predictions_valid = nb_classifier.predict(X_valid_tfidf)

# Evaluate the model on validation data
accuracy_valid = accuracy_score(y_valid, predictions_valid)
print(f'Accuracy on Validation Data: {accuracy_valid}')
print(classification_report(y_valid, predictions_valid, zero_division=1))
print(confusion_matrix(y_valid, predictions_valid))

# Load and preprocess test data
test_essays = pd.read_csv('test_data/test_essays.csv')
test_data = pd.merge(test_essays, train_prompts, on='prompt_id', how='left')
test_data['full_text'] = test_data['prompt_name'] + ' ' + test_data['instructions'] + ' ' + test_data['source_text'] + ' ' + test_data['text']
test_data['full_text'].fillna('', inplace=True)

# Make predictions on test data
X_test_tfidf = tfidf_vectorizer.transform(test_data['full_text'])

# Make predictions on the test data
probabilities = nb_classifier.predict_proba(X_test_tfidf)[:, 1]
submission_df = pd.DataFrame({'id': test_data['id'], 'generated': probabilities})
submission_df.to_csv('submission.csv', index=False)
