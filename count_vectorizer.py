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
    list_of_posts = []

    #iterate through all files in the directory
    for filename in os.listdir(dir):
        print(filename)
        df = pd.read_csv(dir + filename, header=0)

        #get dates and pros columns from data file
        dates = df['date']
        pros = df['pros']

        #get the list of years for all the posts in the file and create a dictionary with years as the keys
        years = [date.split(' ')[3] for date in dates]
        corpus = {key: '' for key in set(years)}

        #add the post to the proper year in the dictionary
        for date, pro in zip(years, pros):
            corpus[date] += pro + ' '

        #sort the keys in order
        keys = sorted(corpus)

        extracted, metric_years = extract_metrics(filename, 'metrics.csv', keys)

        for metric in extracted:
            data.append(metric)
        #only use the metrics from years that present in the datafile
        for key in keys:
            if key in metric_years:
                list_of_posts.append(corpus[key])
    #creates count vectorizer for posts
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    counts = vectorizer.fit_transform(list_of_posts)

    data = np.asarray(data)

    return counts, data


def extract_metrics(filename, metrics, years):
    data = []
    company_name = filename.split('.')[0]
    metric_years = []
    metrics_file = pd.read_csv('metrics.csv')

    # get all the proper metrics from the metrics file
    for row in metrics_file.index:
        # print(metricsFile['Data Year - Fiscal'][row])
        if metrics_file['Company Name'][row] == company_name and str(
                int(metrics_file['Data Year - Fiscal'][row])) in years:
            metric_years.append(str(int(metrics_file['Data Year - Fiscal'][row])))
            # print(str(int(metricsFile['Data Year - Fiscal'][row])))
            data.append(metrics_file['Gross Profit (Loss)'][row])
    return data, metric_years



'''
Random Forest Classifier for the glassdoor posts
'''

def RandomForestModel(X, y):
    clf = RandomForestClassifier(max_depth=5)
    clf.fit(X, y)
    pred = []
    for i in range(8166):
        if i % 2 == 0:
            pred.append(1)
        else:
            pred.append(1)
    print(clf.predict([pred]))

def main():
    X, y = get_counts_and_metrics('C:/Users/Chaitu Konjeti/socweb-glassdoor-project/REVIEWS/')
    RandomForestModel(X, y)

main()