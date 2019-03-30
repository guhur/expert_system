import os
import numpy as np
import configparser
from collections import Counter
import tweepy
import _pickle
import h5py
import gc
import warnings
import joblib

warnings.filterwarnings("ignore", category=UserWarning)


def most_common(lst):
    return max(set(lst), key=lst.count)



def load_offline(str):
    ext = os.path.splitext(str)[1]
    if ext == ".joblib":
        return joblib.load(str)
    else:
        with open(str, 'rb') as f:
            dump = _pickle.load(f)
        return dump


word2index = load_offline('app/static/models/word2index.pkl')
vectorizer = load_offline('app/static/models/vectorizer.pkl')
multinomialnb = load_offline('model/multinomial-nb.joblib')

config = configparser.ConfigParser()
config.read("keys.cfg")
consumer_secret = config.get("Twitter", "consumer_secret")
consumer_token = config.get("Twitter", "consumer_token")
token_key = config.get("Twitter", "token_key")
token_secret = config.get("Twitter", "token_secret")

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(token_key, token_secret)
api = tweepy.API(auth)


def clean(query):
    return vectorizer.transform([query])



def lencode(text):
    vector = []
    for word in text.split(' '):
        try:
            vector.append(word2index[word])
        except KeyError:
            vector.append(0)
    padded_seq = pad_sequences([vector], maxlen=100, value=0.)
    return padded_seq


def word_feats(text):
    return dict([(word, True) for word in text.split(' ')])


def predictor(query):
    clean_query = clean(query)
    mnb = multinomialnb.predict([query])
    return [mnb.tolist()[0]]


def get_most_count(x):
    return Counter(x).most_common()[0][0]


def processing_results(query):
    predict_list = []
    line_sentiment = []
    for t in query:
        p = predictor(t)
        line_sentiment.append(most_common(p))
        predict_list.append(p)

    data = {'MultinomialNB': 0}

    # overal per sentence
    predict_list = np.array(predict_list)
    for i, key in enumerate(data):
        data[key] = get_most_count(predict_list[:, i])

    # all the sentences with 3 emotions
    predict_list = predict_list.tolist()
    emotion_sents = [0, 0, 0]
    for p in predict_list:
        if most_common(p) == 0:
            emotion_sents[0] += 1
        elif most_common(p) == 1:
            emotion_sents[1] += 1
        else:
            emotion_sents[2] += 1

    # overall score
    score = most_common(list(data.values()))
    gc.collect()
    return data, emotion_sents, score, line_sentiment, query, len(query)
