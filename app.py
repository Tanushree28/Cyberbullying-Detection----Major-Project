# with nlpaug with neplai notebook running on RandomForestClassifier
from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2,f_classif, mutual_info_classif

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))


from nlpaug.util import Action
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import logging

logging.basicConfig(filename='app_file_log.log')
# logging.debug('This message should go to the log file')
tweet_input =  st.container()
sentiment_analysis =  st.container()


data = pd.read_csv("src/twitter_nepali.csv")
#print(data.head())

data["class"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive",  3: "Gibberish Undetectable"})
#print(data.head())

data = data[["tweet", "class"]]
#print(data.head())


def decontract(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text
lemmatizer = WordNetLemmatizer()

def process_tweet(text):

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove attherate
    # only removing the hash # sign from the word
    text = re.sub(r'@', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = str(re.sub("\S*\d\S*", "", text).strip())
    text=decontract(text)
    # tokenize texts
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tokens = tokenizer.tokenize(text)

    texts_clean = []
    for word in tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation+'...'):  # remove punctuation
            #
            stem_word = lemmatizer.lemmatize(word,"v")  # Lemmatizing word
            texts_clean.append(stem_word)

    return " ".join(texts_clean)



X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['class'], test_size=0.2,random_state=42)
rf=RandomForestClassifier(n_estimators=10)
# rf.fit(X_train,y_train)
# rf.score(X_test,y_test)

vectorizer = TfidfVectorizer(min_df=3,analyzer='word',max_features=10000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)
rf.fit(X_train_vectorized,y_train)
y_train_pred=rf.predict(X_train_vectorized)



with tweet_input:
    st.header('Is Your Text Considered Cyberbullying?')
    st.write("""*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
    # user input here
    user_text = st.text_area("Enter any Tweet: ")
    if len(user_text) < 1:
        st.write("  ")
    else:
        sample = user_text
        data = vectorizer.transform([sample]).toarray()
        a = rf.predict(data)
        pred = st.button("Predict", key=str)
        st.title(a)

with sentiment_analysis:
    if user_text:
        st.header('Sentiment Analysis with VADER')

        # explaining VADER
        st.write(
            """*VADER is a lexicon designed for scoring social media. More information can be found [here](https://github.com/cjhutto/vaderSentiment).*""")
        # spacer
        st.text('')

        # instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text)
        if sentiment_dict['compound'] >= 0.05:
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05:
            category = ("**Negative 🚫**")
        else:
            category = ("**Neutral ☑️**")

        # score breakdown section with columns
        breakdown, graph = st.columns(2)
        with breakdown:
            # printing category
            st.write("Your Tweet is rated as", category)
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**")
            st.write(sentiment_dict['neg'] * 100, "% Negative")
            st.write(sentiment_dict['neu'] * 100, "% Neutral")
            st.write(sentiment_dict['pos'] * 100, "% Positive")
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph)
            logging.debug('This message should go to the log file')





# streamlit run app.py
# streamlit run streamlit_app.py --logger.level=debug 2>logs.txt

