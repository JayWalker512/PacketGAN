#!/usr/bin/env python3

import random
import torch
import torch.optim as optim 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class KMeansTorch(KMeans):
    def __init__(self, *args, **kw):
        super(KMeansTorch, self).__init__(*args, **kw)
        
    def fit(self, X):
        if isinstance(X, torch.Tensor):
            KMeans.fit(self, X.numpy())
            return
        
        KMeans.fit(self, X)

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            return KMeans.predict(self, X.numpy())
        
        return KMeans.predict(self, X)

def getDictOfLabelCounts(labels):
    labels_counts = {}
    for l in labels:
        if l not in labels_counts:
            labels_counts[l] = 1
        else:
            labels_counts[l] += 1

    return labels_counts

#Need a function that will fit KMeans to a training set, then predict on the test set
#and print bar charts comparing them
def clusterAndCompare(kmeans, train_set, test_set, retrain = True):
    if retrain:
        kmeans.fit(train_set)

    train_labels = kmeans.labels_
    test_labels = kmeans.predict(test_set)
    train_counts = getDictOfLabelCounts(train_labels)
    test_counts = getDictOfLabelCounts(test_labels)

    n_clusters = len(kmeans.cluster_centers_)
    sorted_train_labels = [l for l in range(0, n_clusters)]
    sorted_train_counts = [train_counts[l] for l in sorted_train_labels]
    
    sorted_test_labels = sorted(test_counts.keys())
    sorted_test_counts = [test_counts[l] for l in sorted_test_labels]
    
    bar_width = 0.6
    plt.bar(sorted_train_labels, sorted_train_counts, color='#CCCCCC', label="Train Set", width=bar_width)
    plt.bar(sorted_test_labels, sorted_test_counts, color='#444444', label="Test Set", alpha=0.5, width=bar_width*0.5)
    plt.xticks(sorted_train_labels)
    plt.legend()
    plt.show()

    return train_counts, test_counts


def test_cases():
    X = torch.tensor([[1], [1], [1], [0], [0], [0], [9], [9], [9]])
    kmeans = KMeansTorch(n_clusters = 3)
    kmeans.fit(X)
    y = kmeans.predict(X)
    a = y[0]
    b = y[3]
    c = y[6]
    y_test = np.array([a,a,a,b,b,b,c,c,c])
    for i in range(0,len(y_test)):
        assert y[i] == y_test[i],"Clustering not as expected!"

    train_counts, test_counts = clusterAndCompare(kmeans, X, X)
    for key in train_counts:
        assert train_counts[key] == test_counts[key],"Cluster counts don't match!"

    X_new = [[2], [2], [2], [2], [2], [2], [2], [2], [2]]
    train_counts, test_counts = clusterAndCompare(kmeans, X, X_new)
    cluster = kmeans.predict([[1]])
    for key in test_counts:
        key == cluster[0] #make sure the only cluster matches the prediction for 1
    #print(train_counts)
    #print(test_counts)


if __name__ == "__main__":
    test_cases()
