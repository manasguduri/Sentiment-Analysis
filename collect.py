#from StringIO import StringIO
import json
#import urllib2
import re
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import hashlib
import numpy as np
import csv
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import pickle
from sklearn import neighbors
from zipfile import ZipFile
from sklearn import svm
from pylab import *
import requests
import configparser
from TwitterAPI import TwitterAPI
import sys
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from collections import Counter, defaultdict


twitter = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

twittertweets = []

a = ["thanksgiving -filter:retweets", "cybermonday -filter:retweets", "blackfriday -filter:retweets"]
for item in a:
    for r in twitter.request('search/tweets', {'q': item, 'count': 4000}):
        twittertweets.append(r)

def unicode_text(text):
    string = re.sub('http\S+', 'THIS_IS_A_URL', text)
    tokens = re.findall(r"\w+", string.lower(),flags = re.L)
    tokens1 = []
    for i in tokens:
        x = re.findall(r"\w+", i,flags = re.U)
        for j in x:
            tokens1.append(j)
    return " ".join(tokens1)

for json in twittertweets:
    json['text'] = unicode_text(json['text'])
    json['user']['description'] = unicode_text(json['user']['description'])
    json['user']['name']=unicode_text(json['user']['name'])

        #writer.writerow([json['text']])

tweets = {}
location = {}
user_name = {}
description ={}
hastags = {}
for i in twittertweets:
    hashtag = []
    tweet = []
    if i['user']['screen_name'] in tweets.keys():
        tweets[i['user']['screen_name']].append(i['text'])
    else:
        tweets[i['user']['screen_name']]= [i['text']]
    location[i['user']['screen_name']] = [i['user']['location']]
    user_name[i['user']['screen_name']]= i['user']['name']
    for j in i['entities']['hashtags']:
        hashtag.append(j['text'])
    hastags[i['user']['screen_name']]=hashtag
    description[i['user']['screen_name']] =[i['user']['description']]

data = []
for json in twittertweets:
    tweet_data = []
    tweet_data.append(json['user']['screen_name'])
    tweet_data.append(json['user']['description'])
    tweet_data.append(json['text'])
    tweet_data.append(json['user']['name'])
    data.append(tweet_data)

with open('twitter_data.csv', 'w') as fp:
    a = csv.writer(fp)
    a.writerows(data)

