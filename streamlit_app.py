
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (only once)
# This is added here for completeness, in a real deployment, you might pre-download these
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

# Load the vectorizer and model
vectorizer = joblib.load('count_vectorizer.pkl')
model = joblib.load('logistic_regression_model.pkl')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove punctuation and symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    processed_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(processed_words)

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below to classify its sentiment (positive/negative).')

user_input = st.text_area('Movie Review:', '')

if st.button('Analyze Sentiment'):
    if user_input:
        processed_input = preprocess_text(user_input)
        # Transform the single input using the loaded vectorizer
        input_vectorized = vectorizer.transform([processed_input])
        
        # Predict sentiment
        prediction = model.predict(input_vectorized)
        
        st.write(f'Predicted Sentiment: **{prediction[0].capitalize()}**')
    else:
        st.write('Please enter a review to analyze.')
