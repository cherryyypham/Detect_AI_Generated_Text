# Detecting AI Generated Text

## Overview

This project focuses on the development of a Natural Language Processing (NLP) model for distinguishing between AI-generated and non-AI-generated text. It was created as a submission for the Kaggle competition ["LLM Detect AI-Generated Text"](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview). The project employs various NLP techniques and machine learning algorithms to achieve this distinction.

## Dataset

We used the [DAIGT-V2 dataset]( https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset/discussion) which consists of text samples labeled as either AI-generated or non-AI-generated. This dataset acts as the training data for the model. We split the dataset into training and testing sets, with 80% of the data used for training and 20% for testing for evaluating the model's performance. We submitted the predictions for the test data to Kaggle for evaluation.

## Approach

### 1. **Data Preprocessing**

- **Tokenization and Normalization**: The text data is tokenized using the NLTK library, which involves splitting the text into words. The text is also converted to lowercase for uniformity.
- **Stemming**: PorterStemmer from NLTK is used for stemming, which reduces words to their root form.
- **Stop Words Removal**: Common English stop words are removed to focus on more relevant words in the text.

### 2. **Feature Extraction**

- **TF-IDF Vectorization**: This process is crucial for converting text data into a format that can be used by machine learning algorithms. TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word to a document in a collection of documents. It increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus, which helps to control for the fact that some words are generally more common than others.
- **Max Features**: The TF-IDF vectorizer in this project is set to a maximum of 5000 features. This means it only considers the top 5000 most important words across all documents. This limitation helps in reducing the dimensionality of the feature space, making the model more computationally efficient while still capturing a significant portion of the textual information.
- **Impact on Model Performance**: The choice of features and the way they are extracted can significantly impact the performance of the model. In this case, using TF-IDF ensures that the model does not just focus on the frequency of terms but also how unique these terms are across the documents, which is a valuable characteristic when distinguishing between AI and non-AI text.

### 3. **Model Training**

- **K-Nearest Neighbors (KNN)**: A KNN classifier with 5 neighbors is employed as the machine learning model. This model is chosen for its simplicity and effectiveness in classification tasks.
- **Train-Test Split**: The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing. This split is essential for evaluating the model's performance.

### 4. **Model Evaluation**

- The model's performance is evaluated on the test set using metrics such as precision, recall, and F1-score.

### 5. **Model Persistence**

- The trained KNN model is saved as a pickle file (`knn_model.pkl`) for future use or deployment.

## Requirements

Install the necessary libraries using the following command:

```bash
pip install -r requirements.txt
```

## Usage

Run the Python script to train the model and evaluate its performance. The script will automatically process the text data, train the KNN model, and output the classification report for the test data.

## Limitations and Future Work

- The model's performance is limited by the number of features (5000) considered in TF-IDF vectorization. Exploring a larger feature set or different feature extraction methods could improve performance.
- The choice of the KNN algorithm was based on simplicity. Experimenting with more complex models like neural networks may yield better results.
- Hyperparameter tuning of the KNN model (e.g., number of neighbors) can be explored for optimization.
