import numpy as np
import pandas as pd
import nltk
import re, string
import nltk.data
import subprocess
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Download the required NLTK datasets
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

df_train = pd.read_csv("llm-detect-ai-generated-text/train_essays.csv")
df_test = pd.read_csv("llm-detect-ai-generated-text/test_essays.csv")
df_train_prompts = pd.read_csv("llm-detect-ai-generated-text/train_prompts.csv")
df_train2 = pd.read_csv("train_v2_drcat_02.csv")

# Merge together prompts into training data
merged_df = pd.merge(df_train, df_train_prompts, on="prompt_id", how="left")
merged_df = merged_df.drop(columns=["id", "prompt_id", "instructions", "source_text"])

# Merge together two datasets
df_train2 = df_train2.drop(columns=["source", "RDizzl3_seven"])
df_train2 = df_train2.rename(columns={"label": "generated"})
df = pd.concat([merged_df, df_train2], ignore_index=True)

# Add word count per text
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
print("Not AI Word Count: ", df[df["generated"] == 0]["word_count"].mean())  # Not AI Generated
print("AI Word Count: ", df[df["generated"] == 1]["word_count"].mean())  # AI Generated

# Plot word count distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
train_words_AI = df[df["generated"] == 1]["word_count"]
ax1.hist(train_words_AI, color="red")
ax1.set_title("AI Text")
train_words_not_AI = df[df["generated"] == 0]["word_count"]
ax2.hist(train_words_not_AI, color="green")
ax2.set_title("Not AI Text")
plt.show()

# Look into total characters
df["char_count"] = df["text"].apply(lambda x: len(str(x)))
print("Not AI Character Count: ", df[df["generated"] == 0]["char_count"].mean())  # Not AI
print("AI Character Count: ", df[df["generated"] == 1]["char_count"].mean())  # AI char count


# Define functions for processing text
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.compile("<.*?>").sub("", text)
    text = re.compile("[%s]" % re.escape(string.punctuation)).sub(
        "", text
    )  # Remove square brackets and additional characters
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d", " ", text)  # Replace any digit with a space
    text = re.sub(r"\s+", " ", text)  # Replace any length of whitespace characters with space
    text = text.strip()  # Remove whitespace
    return text


def stopword(text):
    a = [
        i for i in text.split() if i not in stopwords.words("english")
    ]  # Only keep words that aren't stopwords like 'the', 'a', etc.
    return " ".join(a)


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatizer(text):
    word_pos_tags = nltk.pos_tag(word_tokenize(text))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
    return " ".join(a)


def final_process(text):
    return lemmatizer(stopword(preprocess(text)))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array(
            [
                np.mean(
                    [self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)],
                    axis=0,
                )
                for words in X
            ]
        )


# Lemmatize text and clean it
wl = WordNetLemmatizer()  # Converts words to their root: running to run, better to good, etc.
df["clean_text"] = df["text"].apply(lambda x: final_process(x))

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["generated"], test_size=0.2, shuffle=True
)

# Word2Vec model for training and testing data
X_train_tok = [nltk.word_tokenize(i) for i in X_train]
X_test_tok = [nltk.word_tokenize(i) for i in X_test]
df["clean_text_tok"] = [nltk.word_tokenize(i) for i in df["clean_text"]]
model = Word2Vec(df["clean_text_tok"], min_count=1)
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))
modelw = MeanEmbeddingVectorizer(w2v)
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_test_vectors_w2v = modelw.transform(X_test_tok)

# Use Tfidf to vectorize training and testing
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

# FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
lr_tfidf = LogisticRegression(solver="liblinear", C=10, penalty="l2")
lr_tfidf.fit(X_train_vectors_tfidf, y_train)
# Predict y value for test dataset
y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:, 1]
print(classification_report(y_test, y_predict))
print("Confusion Matrix (LR TFIDF):", confusion_matrix(y_test, y_predict))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC (LR TFIDF):", roc_auc)

# FITTING THE CLASSIFICATION MODEL using Logistic Regression (W2v)
lr_w2v = LogisticRegression(solver="liblinear", C=10, penalty="l2")
lr_w2v.fit(X_train_vectors_w2v, y_train)  # model
# Predict y value for test dataset
y_predict = lr_w2v.predict(X_test_vectors_w2v)
y_prob = lr_w2v.predict_proba(X_test_vectors_w2v)[:, 1]
print(classification_report(y_test, y_predict))
print("Confusion Matrix (LR W2V):", confusion_matrix(y_test, y_predict))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC (LR W2V):", roc_auc)
