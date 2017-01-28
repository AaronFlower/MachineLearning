# coding:utf8
# Logistic Regression .

from numpy import *

# 加载数据，取出训练集中的， X, Y。 对于 X 假设特征 x0 = 1.0
def loadData():
	dataMat = []; labelMat = []
	fr = open('./Ch05/testSet.txt')
	for line in fr.readlines():
		lineStr = line.strip()
		lineList = lineStr.split()
		dataMat.append([1.0, float(lineList[0]), float(lineList[1])])
		labelMat.append(float(lineList[2]))
	return dataMat, labelMat

# 对于训练样本 inX 计算出训练样本的假设值。 hypothesis(Z) = 1 / 1 + exp(- W * inX) 
def hypothesis(inX):
	return 1.0 / (1 + exp(-inX)) # 这里的 exp 不 math.exp 而是 numpy 中的 exp 函数，可以处理数组。

# 通过 Batch Gradient Descent 来求出参数 weights. 
# 顺便体验下 python numpy 的强大的矩阵计算功能。
def batchGradientDescent(dataMat, labelMat):
	dataMatX = mat(dataMat)
	dataMatXT = dataMatX.transpose()
	m,n = shape(dataMatX)
	labelMatY = mat(labelMat).transpose()
	weights = ones((n, 1)) 	# 初始化各参数的初始值都为 1.
	alpha = 0.001 					# learning rate, alpha = 0.001
	for i in range(500):
		h = hypothesis(dataMatX * weights)
		error = labelMatY - h
		weights = weights + alpha * dataMatXT *error
	return weights