import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
def data_preparation(data ):
    # Encode the 'label_sexist' column to numerical values
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label_sexist'])

    # Extract tokens and targets based on provided information
    data['token'] = data['token'].fillna('none')
    data['target'] = data['target'].fillna('none')
    return data

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english') and word.isalnum()]

    return ' '.join(tokens)

def remove_noises(text):
    # Convert to lowercase
    text = text.lower()
    noise_pattern = r'\[url\]|\[user\]|\[username\]|\#\w+'
    cleaned_text = re.sub(noise_pattern, '', text)

    return cleaned_text

def data_cleansing(data):
    data['text'] = data['text'].apply(remove_noises)

    return data

def data_preprocess(data):
    # data cleaning
    data = data_cleansing(data)

    # Apply preprocessing to the 'text' column
    data['text_token'] = data['text'].apply(preprocess_text)

    return data

def get_sentiment(text):
    # initialize NLTK sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(text)
    # results in 1 for negative sentences and 0 for positive
    sentiment = 0 if scores['pos'] > 0 else 1

    return sentiment

def extract_feature(data):
    # apply get_sentiment function

    data['sentiment'] = data['text'].apply(get_sentiment)
    return data

def tiuning_hyperparameters(model,X,y):
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(model, param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X, y)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    return  grid.best_params_, grid.best_estimator_


def prediction_with_nltk(data):
    try:

        # preprocess
        df=data_preprocess(data)
        # feature extraction
        df=extract_feature(df)
        # Initialize the TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        X=df['text_token']
        y=df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=77)

        # Fit and transform the text data
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # prediction
        # svm_result = prediction_with_svm(df)
        # Train a Linear SVM model
        svm_model = SVC()
        param,svm_model_=tiuning_hyperparameters(svm_model,X_train_tfidf, y_train)
        svm_model_.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = svm_model_.predict(X_test_tfidf)

        # Evaluate the model
        score = accuracy_score(y_test, y_pred)
        print('Accuracy:', score)
        message = classification_report(y_test, y_pred)
        print(f'Classification Report: \n {message}')

        cfm = confusion_matrix(y_test, y_pred)
        print(cfm)
    except Exception as e:

        message = str(e)
        print(message)

    return score, cfm, message

def get_tfid_data(X,y):
    vectorizer = TfidfVectorizer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform the text data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test


def prediction_with_nltk_NB(data):
    try:

        # preprocess
        df_=data_preprocess(data)
        # feature extraction
        # Initialize the TF-IDF vectorizer
        X = df_['text_token']
        y = df_['label']
        X_train_tfidf, X_test_tfidf, y_train, y_test=get_tfid_data(X, y)
        # prediction

        classifier = MultinomialNB()
        classifier.fit(X_train_tfidf, y_train)

        # Make predictions on the test set
        y_pred = classifier.predict(X_test_tfidf)

        # Evaluate the model
        score = accuracy_score(y_test, y_pred)
        print('Accuracy:', score)
        message = classification_report(y_test, y_pred)
        print(f'Classification Report: \n {message}')

        cfm = confusion_matrix(y_test, y_pred)
        print(cfm)
    except Exception as e:

        message = str(e)
        print(message)

    return score, cfm, message


def prediction_with_svm(data):
    message=''

    # preprocess
    df_ = data_preprocess(data)
    # feature extraction
    # Initialize the TF-IDF vectorizer
    X = df_['text_token']
    y = df_['label']
    X_train_tfidf, X_test_tfidf, y_train, y_test = get_tfid_data(X, y)


    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_tfidf['text'])
    X_test_tfidf = vectorizer.transform(X_test_tfidf['text'])

    try:
        # Train a Linear SVM model
        svm_model = SVC()
        svm_model.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = svm_model.predict(X_test_tfidf)

        # Evaluate the model
        score = accuracy_score(y_test, y_pred)
        print('Accuracy:', score)
        message = classification_report(y_test, y_pred)
        print(f'Classification Report: \n {message}')

        cfm=confusion_matrix(y_test, y_pred)
        print(cfm)
    except Exception as e:
        message=str(e)
    print(message)

    return score, cfm, message

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load data into a DataFrame
    data = pd.read_csv(r"C:\Mahsa's stuff\Aston\Dissertation\Annotation\edos_labelled_aggregated.csv")
    df=data_preparation(data)

    # SVM modeling
    # svm_result=prediction_with_svm(df)

    # nltk modeling
    prediction_with_nltk(data)
    # prediction_with_nltk_NB(data)

