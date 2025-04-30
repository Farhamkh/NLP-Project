from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)