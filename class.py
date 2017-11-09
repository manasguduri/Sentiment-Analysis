import collect
import json
import urllib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
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

def read_tweets(filename):
    tweet_t = []
    user_n = []
    with open(filename, 'r',encoding='utf-8') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in filereader:
            try:
                tweet_t.append(row[2])
                user_n.append(row[3])
            except IndexError:
                pass
    return tweet_t,user_n
tweet_test, user_name = read_tweets('twitter_data.csv')
#print(tweet_test)
def tokenize(text):
    tokens = re.findall(r"\w+|\S", text.lower(),flags = re.L)
    tokens1 = []
    for i in tokens:
        x = re.findall(r"\w+|\S", i,flags = re.U)
        for j in x:
            tokens1.append(j)
    return tokens1
#tweets = open('twitter_data.csv', 'r')
tokens = [tokenize(t) for t in tweet_test]

#testing positive and negative tweets

url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(BytesIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')

afinn = dict()

for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        afinn[parts[0].decode("utf-8")] = int(parts[1])

def afinn_sentiment2(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg

#manually labelling the data
positives = []
negatives = []
all2 = [] 
all1 = []
neutral = []
tweet_manual_labelling = []

for token_list, tweet in zip(tokens, tweet_test):
    pos, neg = afinn_sentiment2(token_list, afinn)
    all2.append((tweet, pos, neg))
    if pos > neg:
        positives.append((tweet, pos, neg))
    elif neg > pos:
        negatives.append((tweet, pos, neg))
    else:
        all1.append((tweet, pos, neg))
for tweet, pos, neg in sorted(positives, key=lambda x: x[1], reverse=True):
    neutral.append((-1, tweet))


for tweet, pos, neg in sorted(negatives, key=lambda x: x[1], reverse=True):
    neutral.append(('1', tweet))

for tweet, pos, neg in sorted(all1, key=lambda x: x[1], reverse=True):
    neutral.append(('0', tweet))


with open('neutral.csv', 'w') as fp1:
    filewriter = csv.writer(fp1)
    filewriter.writerows(neutral)

def read_training_data(filename):
    labeled_tweets = []
    labels1 = []
    with open(filename, 'rt',encoding='utf-8') as csvfile1:
        filereader = csv.reader(csvfile1)
        for row in filereader:
            try:
                labeled_tweets.append(row[1])
                labels1.append(int(row[0]))
            except IndexError:
                pass
    return labeled_tweets,np.array(labels1)

labeled_tweets,labels = read_training_data('neutral.csv')



def do_vectorize(tokenizer_fn=tokenize, min_df=1,
                 max_df=1., binary=True, ngram_range=(1,1)):

    lis = ['thanksgiving','shopping','blackfriday','cybermonday']#list of stop words
    #countvectorizer object
    vectorizer = CountVectorizer(input = 'content', tokenizer = tokenizer_fn, min_df=min_df,
                                     max_df=max_df, binary=binary, ngram_range=ngram_range,
                                 dtype = 'int',analyzer='word',token_pattern='(?u)\b\w\w+\b',encoding='utf-8' )
    return vectorizer

vectorizer = do_vectorize() #get the vectorizer object
matrix = vectorizer.fit_transform(t for t in labeled_tweets) # learn new words and tranform it to a CSR matrix
print ('matrix represents %d documents with %d features' % (matrix.shape[0], matrix.shape[1]))
print('first doc has terms:\n%s' % (str(sorted(matrix[0].nonzero()[1]))))

def repeatable_random(seed):
    hash = str(seed)
    hash=hash.encode('utf-8')
    while True:
        hash = hashlib.md5(hash).digest()
        for c in hash:
            yield c

def repeatable_shuffle(X, y, labeled_tweets):
    r = repeatable_random(42)
    indices = sorted(range(X.shape[0]), key=lambda x: next(r))
    return X[indices], y[indices], np.array(labeled_tweets)[indices]

X, y, twittertweets = repeatable_shuffle(matrix, labels, labeled_tweets)
print(X.toarray())
print('fourth shuffled document %s has label %d and terms: %s' %
      (twittertweets[4], y[4], sorted(X[4].nonzero()[1])))


#Logistic Regression

def get_clf(C = 1.,penalty = 'l2'):
    return LogisticRegression(C = C,penalty=penalty,random_state=42)
clf_logistic = get_clf()

def do_cross_validation(X, y,clf,n_folds):
    cv = KFold(len(y),n_folds)
    print(y)
    #print(cv)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind],y[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(y[test_ind], predictions))
    avg = np.mean(accuracies)
    return avg

logistic_regression_accuracy = (do_cross_validation(X, y,clf_logistic, 5))*100
print('Average cross validation accuracy for Logistic Regression=%.1f percentage' % (logistic_regression_accuracy))

def prediction(CLF,trained_CSR,trained_label,untrained_tweets_CSR):
    CLF.fit(trained_CSR,trained_label)
    predicted = CLF.predict(untrained_tweets_CSR)
    return predicted

vec = do_vectorize()
X = vec.fit_transform(x for x in twittertweets)
test_tweet_vector = vec.transform(t for t in tweet_test)

Tweet_predicted_logistic =prediction(clf_logistic,X,y,test_tweet_vector)

logitic_pred_dict = dict(Counter(Tweet_predicted_logistic))

def print_results(dictionary):
    for i in dictionary:
        if i == -1:
            print ("\tTweets aganist Thanksgiving\t\t%d" %dictionary[i])
        elif i == 0:
            print ("\tNeutral tweets on Thanksgiving\t\t%d" %dictionary[i])
        elif i == 1:
            print ("\tPositive tweets on Thanksgiving\t\t\t%d" %dictionary[i])
        elif i == 2:
            print ("\tTweets supporting Thanksgiving\t%d" %dictionary[i])

print("Logistic Regression Results")
print_results(logitic_pred_dict)

def percentage_tweets(dictionary):
    sumt = sum(dictionary.values())
    d= 0
    for a in sumt:
        d = a+d
       
    percentage = {}
    
    for i in dictionary:
        #print(dictionary[i])
        
        pr = (dictionary[i]*1.0/d)*100
        percentage[i] = pr
    return percentage

logistic_precentage = percentage_tweets(logitic_pred_dict)

def print_results_percentage(dictionary):
    for i in dictionary:
        if i == -1:
            print ("\tTweets aganist Thanksgiving\t\t%.1f percentage" %dictionary[i])
        elif i == 0:
            print ("\tNeutral tweets on Thanksgiving\t\t%.1f percentage" %dictionary[i])
        elif i == 1:
            print ("\tPro Thanksgiving tweets\t\t\t%.1f percentage" %dictionary[i])
        elif i == 2:
            print ("\tTweets supporting Thanksgiving\t%.1f percentage" %dictionary[i])

print("Logistic Regression Percentage")
print_results_percentage(logistic_precentage)

def print_graph(title,dictionary):
    a = dictionary.keys()
    senti = []
    y = [i for i in range(len(a))]
    for x in a:
        if x == -1:
            senti.append('Against')
        elif x == 0:
            senti.append('Neutral')
        elif x == 1:
            senti.append('Positive')
        elif x == 2:
            senti.append('Support')
    values = [dictionary[i] for i in a]

    plt.bar(y,values, align='center')
    plt.xticks(y, senti)
    plt.title(title)
    plt.savefig("LogisticGraph.png")
print_graph('User analysis on Tweet Sentiment by Logistic Regression',logitic_pred_dict)

