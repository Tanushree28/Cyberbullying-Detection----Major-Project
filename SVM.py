# wokin on SVM only with original dataset
import string
import pandas as pd
import numpy as np
import nltk
from textstat.textstat import *
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
import pickle


nltk.download('vader_lexicon')
# from wordcloud import WordCloud
from matplotlib import pyplot as plt
from PIL import Image
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import seaborn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import logging

tweet_input =  st.container()
model_results =  st.container()
sentiment_analysis =  st.container()

hate_speech_df = pd.read_csv("src/twitter.csv")


def preprocess(tweet):
    # removal of extra spaces
    regex_pat = re.compile(r'\s+')
    tweet_space = tweet.str.replace(regex_pat, ' ')

    # removal of mentions (@name)
    regex_pat = re.compile(r'@[\w\-]+')
    tweet_name = tweet_space.str.replace(regex_pat, '')

    # removal of URLd
    giant_url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                                 '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweets = tweet_name.str.replace(giant_url_regex, '')

    # removal of punctuations and numbers
    punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
    # replace whitespace with a single space
    newtweet = punc_remove.str.replace(r'\s+', ' ')
    # remove leading and trailing whitespace
    newtweet = newtweet.str.replace(r'^\s+|\s+?$', '')
    # replace normal numbers with numbr
    newtweet = newtweet.str.replace(r'\d+(\.\d+)?', 'numbr')
    # removal of capitalization
    tweet_lower = newtweet.str.lower()

    # tokenizing
    tokenized_tweet = tweet_lower.apply(lambda x: x.split())

    # removal of stopwords
    tokenized_tweet = tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])

    # stemming of the tweets
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
        tweets_p = tokenized_tweet

    return tweets_p

def tfidf_vectorizer(tweet):
    tweet = pd.Series(tweet)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=1, min_df=1, max_features=10000)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)
    tfidf_vectorizer.fit_transform(hate_speech_df['processed_tweets'])
    tfidf_string = tfidf_vectorizer.transform(tweet)
    tfidf_array = tfidf_string.toarray()
    return tfidf_array


sentiment_analyzer = VS()


def count_tags(tweet_c):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', tweet_c)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def sentiment_analysis_string(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    twitter_objs = count_tags(tweet)
    features = [sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'], twitter_objs[0],
                twitter_objs[1],
                twitter_objs[2]]
    return features


def sentiment_analysis_array(tweets):
    features = []
    for t in tweets:
        features.append(sentiment_analysis_string(t))
    return np.array(features)


from textstat.textstat import *


def additional_features(tweet):
    syllables = textstat.syllable_count(tweet)
    num_chars = sum(len(w) for w in tweet)
    num_chars_total = len(tweet)
    num_words = len(tweet.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(tweet.split()))

    # Modified FKRA grade
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    # Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    add_features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_words,
                    num_unique_terms]
    return add_features


def get_additonal_feature_array(tweets):
    features = []
    for t in tweets:
        features.append(additional_features(t))
    return np.array(features)


with tweet_input:
    st.header('Is Your Text Considered Cyberbullying?')
    st.write(
        """*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
    # user input here
    user_text = st.text_area("Enter any Tweet: ")
    if user_text:
        def get_predictions(tweet):
            # Convert the string to a panda serie in order to apply the following functions
            tweet = pd.Series(tweet)
            # Get the sentiment analysis of the un-preprocessed string
            # we need to apply this function when the string is not yet pre processed in order to keep the whole meaning
            # of the sentence, like the # for exemple
            array_sentiment_analysis = sentiment_analysis_array(tweet)
            # Now that we have extract the sentiment from the sentence, let pre-process our tweet
            preprocessed_tweet = preprocess(tweet)
            # We convert our string into a matrix
            array_tfidf = tfidf_vectorizer(tweet)
            # Add additional featutre
            additional_features = get_additonal_feature_array(tweet)
            # Concatenate all the features
            features_tweet_test = np.concatenate([array_tfidf, array_sentiment_analysis, additional_features], axis=1)
            # Transform our array to a dataframe
            df = pd.DataFrame(features_tweet_test)
            # df = df.iloc[:, :1].values
            # We apply our model to our tweet
            final_model = pickle.load(open('final_model.pkl', 'rb'))
            pred = final_model.predict(df)

with model_results:
    pred = st.button("Predict", key = str)
    st.subheader('Prediction:')
    st.write(pred)
    if pred == 0:
        # st.write(pred)
        st.subheader('**Hate Speech**')
    elif pred == 1:
        # st.write(pred)
        st.subheader('**Offensive Language**')
    else:
        # st.write(pred)
        st.subheader('**Neither**')


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
            category = ("**Positive ???**")
        elif sentiment_dict['compound'] <= - 0.05:
            category = ("**Negative ????**")
        else:
            category = ("**Neutral ??????**")

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

# streamlit run SVM.py