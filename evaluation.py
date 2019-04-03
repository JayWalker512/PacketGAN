#!/usr/bin/env python3

import random
import torch
import torch.optim as optim 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import networks
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

#This function calculates the "Wasserstein Critic" metric described in the paper titled 
#"Pros and cons of GAN evaluation measures". This is an approximation to the Wasserstein Distance
#since that cannot be directly computed in this case. The value of this metric is that
#it addresses both overfitting and mode collapse. A value close to 0 is good, as it means
#the critic cannot distinguish between real and fake samples. A value close to 1 is bad,
#as it means they are easily distinguishable.
#
#This method expects an array/list/DataSet of examples where each example has shape
#(batch_size, sequence_length, num_features).
#
#This implementation expects sequences. The algorithm in general can work on any data
#(not just sequential) but this implementation maps sequences into a lower dimensional 
#latent space that is not required for non-sequential data.
#
#returns a tuple of (wasserstein distance, classification accuracy)
def wasserstein_critic(real_train_set, real_test_set, fake_train_set, fake_test_set):
    #TODO FIXME weights in classifier need to be clipped to some fixed value
    #so that it has a maximum derivative (lipschitz continuous function).

    #TODO only need real_set and fake_set as inputs. 
    #Can do the test set splitting inside this function.

    #First validate that all the data provided are compatible for this test...
    batch_size = real_train_set[0].shape[0]
    sequence_length = real_train_set[0].shape[1]
    num_features = real_train_set[0].shape[2]
    for s in [real_test_set, fake_train_set, fake_test_set]:
        for e in s:
            assert e.shape[0] == batch_size,"Batch size mismatch."
            assert e.shape[1] == sequence_length,"Sequence length mismatch."
            assert e.shape[2] == num_features,"Feature shape mismatch."

    #train set lengths should match each other, but don't need to be same as test sets
    #though they should match themselves as well.
    assert len(real_train_set) == len(fake_train_set),"Training sets of unequal length."
    assert len(real_test_set) == len(fake_test_set),"Testing sets of unequal length."

    latent_size = math.ceil(num_features / 10) #ensure it's at least 1-dimensional
    lsm = networks.GRUMapping(num_features, latent_size)
    clf = networks.NeuralClassifier(latent_size, 1) #outputs binary classification
    criterion = nn.MSELoss()
    optimizer = optim.SGD(clf.parameters(), lr=0.1)

    #Train the classifier for... one epoch? Maybe this is enough? Don't want to overfit...
    #Really it depends on how many examples there are and a lot of other factors, so this
    #is a bit of a "magic number"...

    #Let's make it so the number of training steps is always at least 2500
    #regardless of how many training examples there are.
    #this should be enough to see meaningful progress.
    c = 2500
    n_examples = len(real_train_set)
    n_epochs = math.ceil(c/n_examples) 
    for e in range(0, n_epochs):
        for i in range(0, n_examples):
            clf.zero_grad()
            prediction = clf(lsm(real_train_set[i]).detach())
            loss = criterion(prediction, torch.ones(1))
            loss.backward()
            optimizer.step()
            
            clf.zero_grad()
            prediction = clf(lsm(fake_train_set[i]).detach())
            loss = criterion(prediction, torch.zeros(1))
            loss.backward()
            optimizer.step()

    real_classification_sum = 0
    fake_classification_sum = 0
    N = len(real_test_set) #N is length of EACH set
    n_correct = 0
    for i in range(0, len(real_test_set)):
        with torch.no_grad(): #Don't need to backpropogate
            prediction = clf(lsm(real_test_set[i])).item()
            real_classification_sum += prediction
            if prediction >= 0.5:
                n_correct += 1

            prediction = clf(lsm(fake_test_set[i])).item()
            fake_classification_sum += prediction
            if prediction < 0.5:
                n_correct += 1
    
    w_hat = ((1.0/N)*real_classification_sum) - ((1.0/N)*fake_classification_sum)
    return w_hat, (n_correct/(N*2))




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
