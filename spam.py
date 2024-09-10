import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from nltk.corpus import stopwords
import nltk
import streamlit as st

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('C:/Users/SARTHAK/Desktop/byte/Task_1.csv')

param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'solver': ['liblinear', 'lbfgs'],  # Solvers
        'penalty': ['l2']  # Penalty (L2 Ridge regularization)
    }

# Language detection function
def detect_language(text):
    try:
        return detect(text)
    except:
        return None

# Detect languages and preprocess data

# Streamlit interface
st.title("Spam Prediction")
choice = st.text_input('Choose language, 1 for English, 2 for German, 3 for French: ')

# Initialize df1
df1 = None

# Select dataset based on user choice
if choice == '1':
    df1 = pd.read_csv('C:/Users/SARTHAK/Desktop/byte/english.csv')
elif choice == '2':
    df1 = pd.read_csv('C:/Users/SARTHAK/Desktop/byte/german.csv')
elif choice == '3':
    df1 = pd.read_csv('C:/Users/SARTHAK/Desktop/byte/french.csv')
else:
    st.write('Not valid')
    st.stop()

# Check if df1 is defined
if df1 is not None:
    Y = df1['labels']
    X = df1['text']

    # Create feature extraction object based on language
    first_row = df1.iloc[1]
    first_row['language'] = detect_language(first_row['text'])

    # Define stop words for different languages
    english_stop_words = 'english'
    german_stop_words = stopwords.words('german')
    french_stop_words = stopwords.words('french')

    # Determine the feature extraction settings based on language
    if first_row['language'] == 'en':
        feature_extraction = TfidfVectorizer(min_df=1, stop_words=english_stop_words, lowercase=True)
    elif first_row['language'] == 'de':
        feature_extraction = TfidfVectorizer(min_df=1, stop_words=german_stop_words, lowercase=True)
    elif first_row['language'] == 'fr':
        feature_extraction = TfidfVectorizer(min_df=1, stop_words=french_stop_words, lowercase=True)
    else:
        feature_extraction = TfidfVectorizer(min_df=1, stop_words=None, lowercase=True)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Transform text data
    X_train = feature_extraction.fit_transform(X_train)
    X_test = feature_extraction.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter = 1000)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train , Y_train)

    # Evaluate model
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)
    st.write(f'Accuracy score is: {test_accuracy}')

    # Prediction for user input
    line = st.text_input('Add a sentence')
    if line:
        vectorized_sentence = feature_extraction.transform([line])
        prediction = best_model.predict(vectorized_sentence)
        prediction = int(prediction)  # Ensure prediction is an int
        if prediction == 1:
            st.write('This is ham')
        else:
            st.write('This is spam')
else:
    st.write('No data available for the selected language.')

