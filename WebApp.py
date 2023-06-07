# Import the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array

from keras.preprocessing.text import one_hot, Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model as load_keras_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM

# Download the NLTK stopwords
nltk.download('stopwords')

# Function to preprocess the text
def preprocess_text(sen):
    sentence = sen.lower()
    sentence = re.sub('<[^>]+>', ' ', sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    return sentence

# Function to load and preprocess the data
def load_data():
    movie_reviews = pd.read_csv('a1_IMDB_Dataset.csv')
    X = []
    sentences = list(movie_reviews['review'])
    for sen in sentences:
        X.append(preprocess_text(sen))
    y = movie_reviews['sentiment']
    y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
    return X, y

# Load the trained model
def load_custom_model():
    model = load_keras_model('lstm_model.h5')
    return model

# Function to predict sentiment
def predict_sentiment(review, model):
    review = preprocess_text(review)
    review = [review]
    review = word_tokenizer.texts_to_sequences(review)
    review = pad_sequences(review, padding='post', maxlen=maxlen)
    prediction = model.predict(review)
    return prediction[0][0]

# Load the data
X, y = load_data()

# Tokenize the data
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X)

# Define the maximum length
maxlen = 100

# Load the model
model = load_custom_model()

# Create the Streamlit web app
def main():
    st.title("Sentiment Analysis using LSTM")
    review = st.text_area("Enter a movie review", height=100)
    if st.button("Predict"):
        prediction = predict_sentiment(review, model)
        if prediction > 0.5:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

if __name__ == '__main__':
    main()
