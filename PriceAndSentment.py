import os.path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

import numpy as np

import json
import requests
import time
import datetime as dt
import decimal
import re, string, random
import pandas as pd
import csv

import nltk
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def convert_to_tic(s):
    return time.mktime(dt.datetime.strptime('{:.19}'.format(s), "%Y-%m-%dT%H:%M:%S").timetuple())

def prepare_SA_Feed():
    #Unix date corresponding to 2015-06-15
    initDate=convert_to_tic("2015-06-15T00:00:00")
    # 2020-06-14
    endDate=convert_to_tic("2020-06-13T00:00:00")
    seekingAlphaUrlMask = "https://seekingalpha.com/api/v3/symbols/wfc/news?filter[until]={0}&id=wfc&include=author,primaryTickers,secondaryTickers,sentiments&isMounting=false&page[size]=25"

    # 15 days range
    incrementTimeFrameTicks = 86400*15
    lastTimeTicks = initDate
    if(not os.path.exists('./data/SA_Feed.csv')):
        newsFeed=open('./data/SA_Feed.csv', 'w+')
        while lastTimeTicks < endDate:
            seekingAlphaUrl=str.format(seekingAlphaUrlMask, lastTimeTicks)
            response=requests.get(seekingAlphaUrl)
            if(not response.ok):
                print("Bad")
            else:
                jdata = json.loads(response.content)
                
                for jDataRow in jdata['data']:
                    timeTicksPublished= convert_to_tic(jDataRow['attributes']['publishOn'])
                    feedRow=str(dt.datetime.fromtimestamp(timeTicksPublished))+";"+jDataRow['id'] + ";" +jDataRow['attributes']['title']+ ";"+'\n'
                    newsFeed.write(feedRow)
                    if(timeTicksPublished>lastTimeTicks):              
                        lastTimeTicks=timeTicksPublished
                
                print('Finished requesting SeekingAlpha time period till: ', dt.datetime.fromtimestamp(lastTimeTicks))
                lastTimeTicks+=incrementTimeFrameTicks
                

        newsFeed.close()

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def prepare_tweets_for_model(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tokens)


def prepare_sentiment_classifier():
    if(not os.path.exists('./.venv/nltk_data')):
        os.makedirs('./.venv/nltk_data')
        nltk.download('twitter_samples', './.venv/nltk_data/')
        nltk.download('punkt', './.venv/nltk_data/')
        nltk.download('averaged_perceptron_tagger','./.venv/nltk_data/')
        nltk.download('wordnet','./.venv/nltk_data/')
        nltk.download('stopwords','./.venv/nltk_data/')

    #Normalization - canonical form conversion

    #Define a lexical tags in the tweets and lemmatize (remove the past form, ending etc.)
    #Then remove noise and stop words
    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    # List the most used positive words
    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    # Prepare a dictionary for Bayes
    positive_tokens_for_model = prepare_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = prepare_tweets_for_model(negative_cleaned_tokens_list)

    negative_sentiment_value = -10
    positive_sentiment_value = 10
    positive_dataset = [(tweet_dict, positive_sentiment_value)
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, negative_sentiment_value)
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset
    #Mix the data to avoid bias
    random.shuffle(dataset)

    train_data = dataset[:7000]

    #The rest 3k of 10k tweets are for testing
    test_data = dataset[7000:]
    #Train the model
    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    #print(classifier.show_most_informative_features(10))
    return classifier

if __name__ == "__main__":

    print('Starting Wells Fargo sentiment and price calculation')
    print('Preparing a sentiment classifier')
    bayesClassifier = prepare_sentiment_classifier()
    
    print('Preparing a SeekingAlpha news feed titles for Wells Fargo')
    prepare_SA_Feed()

    saFeedDataSet={}

    print('Parsing a SeekingAlpha news feed titles for Wells Fargo')
    #Get the SeekingAlpha news titles
    
    x1 = random.randint(1, 100)
    x2 = random.randint(100, 200)
    x3 = random.randint(200, 300)
    randNews=[x1,x2,x3]
    with open('./data/SA_Feed.csv', 'r+') as saFeedFile:
        reader = csv.reader(saFeedFile, delimiter=';')
        for index, row in enumerate(reader):
            cleanedTitleRow = remove_noise(word_tokenize(row[2]))
            sentiment = bayesClassifier.classify(dict([token, True] for token in cleanedTitleRow))
            date=dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            if(index in randNews):
                print(row[2]+' has a sentiment: {0}'.format(sentiment))
            saFeedDataSet[date] = sentiment
    
    lists = list(saFeedDataSet.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    print('Parsing Yahoo finance historical prices for Wells Fargo')
    priceDF={}
    with open('./data/WFC_PriceHistData.csv') as f:
        reader=csv.reader(f, delimiter=',')
        next(f)
        for row in reader:
            date=dt.datetime.strptime(row[0], '%Y-%m-%d')
            priceDF[date]=decimal.Decimal(row[1])

    listPrice = list(priceDF.items()) # sorted by key, return a list of tuples

    xp, yp = zip(*listPrice)

    print('Creating a plot for opening WFC price and news sentiments for the given time range (negative=-10, positive=+10)')
    fig, ax = plt.subplots()
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    plt.xlabel('Year', fontsize=20)

    plt.bar(x, y)
    plt.plot(xp,yp)

    plt.show()

    
