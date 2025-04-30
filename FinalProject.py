from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

X_clean = X.apply(preprocess_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

# testing the preprocessing function
#print(X_train.iloc[0])
#print(X_test.iloc[0])
#print(y_train.iloc[0])
#print(y_test.iloc[0])


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


