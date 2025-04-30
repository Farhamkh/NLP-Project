from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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