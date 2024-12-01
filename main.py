from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from urllib.parse import urlparse
import re
import pandas as pd

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize PorterStemmer
ps = PorterStemmer()

# Load or create the vectorizer
def get_or_fit_vectorizer():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            check_is_fitted(vectorizer)
            return vectorizer
    except Exception:
        # Example training corpus to fit the vectorizer
        training_corpus = [
            "This is a spam message",
            "This is a ham message",
            "Spam messages are annoying",
            "Ham messages are helpful"
        ]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(training_corpus)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        return vectorizer

tfidf = get_or_fit_vectorizer()

# Load the spam detection model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    model = None

# Load phishing model (optional)
try:
    phishing_model = pickle.load(open('phishing_url.pkl', 'rb'))
except FileNotFoundError:
    phishing_model = None

# Text preprocessing function for spam detection
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# Function to preprocess URLs for phishing detection
def preprocess_url(url):
    parsed_url = urlparse(url)
    features = {
        "Length_of_URL": len(url),
        "Have_IP": 1 if re.match(r"(\d{1,3}\.){3}\d{1,3}", parsed_url.netloc) else 0,
        "Have_At": 1 if "@" in url else 0,
        "Number_of_Dots": url.count("."),
        "Number_of_Special_Char": len(re.findall(r"[\W_]", url)) - url.count("."),
        "Length_of_Domain": len(parsed_url.netloc),
        "Have_Https": 1 if parsed_url.scheme == "https" else 0,
    }
    return pd.DataFrame([features])

# Function to predict phishing URLs
def predict_phishing(url):
    if phishing_model is None:
        return None, "⚠️ Phishing detection model not loaded."
    features = preprocess_url(url)
    prediction = phishing_model.predict(features)[0]
    probability = phishing_model.predict_proba(features)[0][1]
    return prediction, probability

# Function to detect malware
def detect_malware(file):
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension in ["exe", "dll", "zip", "tar", "pdf"]:
        return "🚨 Malware detected! Suspicious file type."
    return "✅ No malware detected for this file."

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_sms = request.form.get('input_sms')
        input_url = request.form.get('input_url')
        uploaded_file = request.files.get('uploaded_file')
        malware_file = request.files.get('malware_file')

        results = {}

        # Spam Detection
        if input_sms:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            results['spam_result'] = "🚨 Spam Detected!" if result == 1 else "✅ Not Spam"

        # Phishing URL Detection
        if input_url:
            phishing_result, phishing_probability = predict_phishing(input_url)
            if phishing_result is None:
                results['phishing_result'] = phishing_probability
            else:
                results['phishing_result'] = f"🚨 Phishing URL Detected ({phishing_probability:.2%})" if phishing_result == 1 else f"✅ Safe URL ({1 - phishing_probability:.2%})"

        # File-based Spam Detection