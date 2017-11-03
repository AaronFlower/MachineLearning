# encoding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadData(file):
    ''' load data from file '''
    data = np.loadtxt(file)
    features = data[:, 0:-1]
    labels = data[:, -1]
    return features, labels

def regularize(X):
    xMean = X.mean(axis=0)
    xVar = X.var(axis=0)
    return (X - xMean)/xVar
    
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    yMean = yArr.mean()
    y = yArr - yMean
    yMat = np.mat(y[:, np.newaxis])
    xMat = regularize(xArr)
    numSamples, numFeatures = xMat.shape
    ws = np.mat(np.zeros((numFeatures, 1)))
    wsMax = ws.copy()
    for i in range(numIt):
      lowestError = np.inf
      for j in range(numFeatures):
        for sign in [-1, 1]:
          wsTest = ws.copy()
          wsTest[j] += eps * sign
          yTest = xMat * wsTest
          rssE = rssError(yMat.A, yTest.A)
          if rssE < lowestError:
            lowestError = rssE
            wsMax = wsTest.copy()
      ws = wsMax
    print ('lowestError:', lowestError)
    return ws

def main():
    features, labels = loadData('Ch08/abalone.txt')
    weights = stageWise(features, labels, 0.001, 5000)
    print('Max weights:', weights.T)

if __name__ == '__main__':
    main()
