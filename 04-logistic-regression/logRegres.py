#coding:utf-8
from numpy import *

# 加载数据集
def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('Ch05/testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

# sigmoid 函数
def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))

# 梯度上升算法, 求出最优系数.
def gradientAscent(dataMatIn, classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	m, n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights
