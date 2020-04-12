from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
import numpy as np
import csv
import os
import pandas as pd

'''
Separates the posts contained in the provided file by year and then returns a list containing all the counts
'''


def get_counts_and_metrics(dir):
    data = []
    for filename in os.listdir(dir):
        print(filename)
        df = pd.read_csv(dir + filename, header=0)
        dates = df['date']
        pros = df['pros']
        years = [date.split(' ')[3] for date in dates]
        corpus = {key: '' for key in set(years)}
        for date, pro in zip(years, pros):
            corpus[date] += pro + ' '


        company_name = filename.split('.')[0]

        metricsFile = pd.read_csv('metrics.csv')
        rowNum = 0
        for name in metricsFile['Company Name']:
            if name == company_name:
                data.append(metricsFile['Gross Profit (Loss)'][rowNum])
            rowNum += 1


    print(corpus)
    keys = sorted(corpus)
    list_values = []
    for key in keys:
        list_values.append(corpus[key])
    print(list_values)
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    counts = vectorizer.fit_transform(list_values)

    # #metrics
    # company_name = filename.split('.')[0]
    #
    # metricsFile = pd.read_csv('metrics.csv')
    # rowNum = 0
    # for name in metricsFile['Company Name']:
    #     if name == company_name:
    #         data.append(metricsFile['Gross Profit (Loss)'][rowNum])
    #     rowNum += 1
    return counts, data


def extract_metrics(company_name, metrics):
    data = []
    metricsFile = pd.read_csv(metrics)
    rowNum = 0
    for name in metricsFile['Company Name']:
        if name == company_name:
            data.append(metricsFile['Gross Profit (Loss)'][rowNum])
        rowNum += 1
    return data

### NEED TO DO:
### Write code that gets the performance measures from the metrics excel sheet
### The performance measures we are using are: Revenue, Net Income, Gross Profit, and Sales


'''
Random Forest Classifier for the glassdoor posts
'''


def RandomForestModel(X, y):
    clf = RandomForestClassifier(max_depth=5)
    clf = LinearRegression()
    clf.fit(X, y)
    pred = []
    for i in range(1456):
        if i % 2 == 0:
            pred.append(1)
        else:
            pred.append(1)
    print(clf.predict([pred]))

counts, y = get_counts_and_metrics('C:/Users/Chaitu Konjeti/socweb-glassdoor-project/REVIEWS/')
#print(counts)
#y = extract_metrics('AMERICAN AIRLINES GROUP INC', 'metrics.csv')
y = np.asarray(y)
#print(counts)
print(counts.shape)
print(y.shape)
RandomForestModel(counts, y)
