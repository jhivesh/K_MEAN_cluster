import math  # For pow and sqrt
import sys
from random import shuffle, uniform
import numpy as np
import matplotlib.pyplot as plt


###_Pre-Processing_###
from sklearn.cluster import KMeans


def ReadData(fileName):
    f = open("fileName", "r")
    lines = f.read().splitlines()
    f.close()

    item = []

    for i in range(1, len(lines)):
        line = lines[i].split(',')
        itemFeatures = []

        for j in range(len(line) - 1):
            v = float(line[j])  # Convert feature value to float
            itemFeatures.append(v)  # Add feature value to dict

        items.append(itemFeatures)

    shuffle(items)

    return items

###_Auxiliary Function_###
def FindColMinMax(items):
    n = len(items[0])
    minima = [sys.maxsize for i in range(n)]
    maxima = [-sys.maxsize - 1 for i in range(n)]

    for item in items:
        for f in range(len(item)):
            if (item[f] < minima[f]):
                minima[f] = item[f]

            if (item[f] > maxima[f]):
                maxima[f] = item[f]

    return minima, maxima


def EuclideanDistance(x, y):
    S = 0  # The sum of the squared differences of the elements
    for i in range(len(x)):
        S += math.pow(x[i] - y[i], 2)

    return math.sqrt(S)  # The square root of the sum



def InitializeMean(items, k, cMin, cMax):
    # Initialize means to random numbers between
    # the min and max of each column/feature

    f = len(items[0])     # the number of features
    means = [ [ 0 for i in range(f) for j in range(k)]]

    for mean in means:
        for i in range(len(mean)):
            # Set value to a random float
            # (adding +-1 to avoid a wide placement of a mean)
            mean[i] = uniform(cMin[i] + 1, cMax[i] - 1)

    return means


def UpdateMean(n, mean, item):
    for i in range(len(mean)):
        m = mean[i]
        m = (m * (n - 1) + item[i]) / float(n)
        mean[i] = round(m, 3)

    return mean


def FindClusters(means, items):
    clusters = [[] for i in range(len(means))]  # Init clusters

    for item in items:
        # Classify item into a cluster
        index = Classify(means, item)

        # Add item to cluster
        clusters[index].append(item)

    return clusters



###_Core Functions_###
def Classify(means, item):
    # Classify item to the mean with minimum distance

    minimum = sys.maxsize
    index = -1

    for i in range(len(means)):
        # Find distance from item to mean
        dis = EuclideanDistance(item, means[i])

        if (dis < minimum):
            minimum = dis
            index = i

    return index
