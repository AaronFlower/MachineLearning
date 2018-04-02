#!/usr/bin/env python
#coding:utf-8
# simple linear regression

# Packages
# Pandas 用来读取 csv 文件
# Numpy 用来操作数组和矩阵
# Matplotlib 用来画图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据, 取出数据集中的 X, y; 并且假设 x0 = 1
def loadData():
    xMat = []
    y = []
    fr = open('/food-trunk-dataset.txt')
    for line in fr.readlines():
        data = line.strip().split()
        xMat.append([1.0, float(data[0])])
        y.append(float(data[1]))

    return xMat, y


# 用 pandas 读取 csv 文件
def loadDataFromCsv():
    data = pd.read_csv('./foot-trunk-dataset.txt', names = ['population', 'profit'])
    return data
# 生成预处理后的数据
def getProcessedData():
    data = loadDataFromCsv()
    X_df = pd.DataFrame(data.population)
    y_df = pd.DataFrame(data.profit)
    X_df.insert(0, 'intercept', 1.0)
    xMat = np.array(X_df)
    y = np.array(y_df).flatten()
    return xMat, y

# 绘制基本数据分布
def plotData(data):
    x_df = pd.DataFrame(data.population)
    y_df = pd.DataFrame(data.profit)
    plt.figure(figsize = (10, 8))
    plt.plot(x_df, y_df, 'kx')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

# batch gradient descent
def batchGradientDescent(xMat, y, weights, iterations, alpha):
		for i in range(iterations):
			loss = xMat.dot(weights) - y
			gradients = xMat.T.dot(loss)
			weights = weights - alpha * gradients / y.shape[0]
			# print("itreation %d ", i)
			# print(gradients)
			# print(weights)
		return weights

xMat, y = getProcessedData()

iterations = 1000
alpha = 0.01
ws = batchGradientDescent(xMat, y, np.array([0.0, 0.0]), iterations, alpha)
print(iterations, alpha, ws)

iterations = 1500
alpha = 0.01
ws = batchGradientDescent(xMat, y, np.array([0.0, 0.0]), iterations, alpha)
print(iterations, alpha, ws)


