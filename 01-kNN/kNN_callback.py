import operator
from numpy import *

def createDataSet():
    group = array([[1, 1.1], [1,1], [0,0],[0,0.1]])
    labels = ['A','A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistanceIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        votedLabel = labels[sortedDistanceIndicies[i]]
        classCount[votedLabel] = classCount.get(votedLabel, 0) + 1
    
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0]




