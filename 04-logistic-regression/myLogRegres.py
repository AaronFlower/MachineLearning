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

# 求出 weights 参数后，我们可以根据 weight 来画出拟合曲线。
'''
	在本例中，因为假设 Z = w0 * x0 + w1 * x1 + w2 * x2
	其中 x0 = 1, 我们把 x1 特征画在 x 轴上， x2 特征画在 y 轴上。
	所以  y = -(w0 + w1 * x) / w2
'''
def plotBestFit(weights):
	import matplotlib.pyplot as plt
	dataMat, labelMat = loadData()
	numOfData = len(dataMat)
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(numOfData):
		if labelMat[i] == 1:
			xcord1.append(dataMat[i][1])
			ycord1.append(dataMat[i][2])
		else:
			xcord2.append(dataMat[i][1])
			ycord2.append(dataMat[i][2])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='r', marker='s')
	ax.scatter(xcord2, ycord2, s=30)
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()