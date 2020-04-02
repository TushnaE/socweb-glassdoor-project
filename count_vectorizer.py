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
    for filename in os.listdir(dir):
        print(filename)
        corpus = {}
        with open('REVIEWS/' + filename) as f:
            csv_reader = csv.reader(f, delimiter=',')
            count = 0
            for row in csv_reader:
                if count == 0:
                    count += 1
                    continue
                else:
                    try:
                        year = row[0].split(' ')[3]
                        if year not in corpus.keys():
                            string = ''
                            string += row[7]
                            corpus[year] = string + ' '
                        else:
                            corpus[year] += row[7] + ' '
                        count += 1
                    except:
                        count += 1
        keys = sorted(corpus)
        list_values = []
        for key in keys:
            list_values.append(corpus[key])
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
        counts = vectorizer.fit_transform(list_values).toarray()
        return counts

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


counts = get_counts_and_metrics('C:/Users/Chaitu Konjeti/Glassdoor/REVIEWS/')

RandomForestModel(counts, y)
