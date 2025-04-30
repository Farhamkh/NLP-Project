from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('test.csv')

df['text'] = df['Title'].fillna('') + " " + df['Description'].fillna('')
df['text'] = df['text'].astype(str)


X = df['text']
y = df['Class Index']

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatization with Lowercasing
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in filtered_tokens]

    # Punctuation Removal
    cleaned_tokens = [
        re.sub(f"[{re.escape(string.punctuation)}]", "", word)
        for word in lemmatized_tokens
    ]
    cleaned_tokens = [word for word in cleaned_tokens if word]

    return " ".join([word for word in cleaned_tokens if word])

def preprocess_text_stemming(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Stemming with Lowercasing
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word.lower()) for word in filtered_tokens]

    # Punctuation Removal
    cleaned_tokens = [
        re.sub(f"[{re.escape(string.punctuation)}]", "", word)
        for word in stemmed_tokens
    ]
    cleaned_tokens = [word for word in cleaned_tokens if word]

    return " ".join(cleaned_tokens)

X_clean = X.apply(preprocess_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train LinearSVC
print("Training SVM...")
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
svm_preds = svm_model.predict(X_test_tfidf)

# Train Logistic Regression
print("Training Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf, y_train)
log_preds = log_model.predict(X_test_tfidf)

# Evaluate SVM
print("Evaluating SVM...")
print(classification_report(y_test, svm_preds))
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))

# Evaluate Logistic Regression
print("Evaluating Logistic Regression...")
print(classification_report(y_test, log_preds))
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))

# Confusion matrix (SVM)
cm_svm = confusion_matrix(y_test, svm_preds)
plt.figure(figsize=(8,6))
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot(cmap=plt.cm.Blues)
plt.title("SVM Confusion Matrix")
plt.show()

# Confusion matrix (Logistic Regression)
cm_log = confusion_matrix(y_test, log_preds)
plt.figure(figsize=(8,6))
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp_log.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# Stemming + CountVectorizer
X_clean_stem = X.apply(preprocess_text_stemming)

# Train/test split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_clean_stem, y, test_size=0.2, random_state=42)

# CountVectorizer
print("Vectorizing with CountVectorizer...")
count_vect = CountVectorizer()
X_train_s_vect = count_vect.fit_transform(X_train_s)
X_test_s_vect = count_vect.transform(X_test_s)

# Train and Evaluate SVM
print("Training SVM with Stemming...")
svm_model_s = LinearSVC()
svm_model_s.fit(X_train_s_vect, y_train_s)
svm_preds_s = svm_model_s.predict(X_test_s_vect)

# Train and Evaluate Logistic Regression
print("Training Logistic Regression with Stemming...")
log_model_s = LogisticRegression(max_iter=1000)
log_model_s.fit(X_train_s_vect, y_train_s)
log_preds_s = log_model_s.predict(X_test_s_vect)

# Evaluate SVM with Stemming
print("Evaluating SVM with Stemming...")
print(classification_report(y_test_s, svm_preds_s))
print("SVM with Stemming Accuracy:", accuracy_score(y_test_s, svm_preds_s))

# Evaluate Logistic Regression with Stemming
print("Evaluating Logistic Regression with Stemming...")
print(classification_report(y_test_s, log_preds_s))
print("Logistic Regression with Stemming Accuracy:", accuracy_score(y_test_s, log_preds_s))

# Confusion matrix (SVM with Stemming)
cm_svm_s = confusion_matrix(y_test_s, svm_preds_s)
plt.figure(figsize=(8,6))
disp_svm_s = ConfusionMatrixDisplay(confusion_matrix=cm_svm_s)
disp_svm_s.plot(cmap=plt.cm.Blues)
plt.title("SVM with Stemming Confusion Matrix")
plt.show()

# Confusion matrix (Logistic Regression with Stemming)
cm_log_s = confusion_matrix(y_test_s, log_preds_s)
plt.figure(figsize=(8,6))
disp_log_s = ConfusionMatrixDisplay(confusion_matrix=cm_log_s)
disp_log_s.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression with Stemming Confusion Matrix")
plt.show()
