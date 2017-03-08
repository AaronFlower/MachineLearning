# coding: utf8

from numpy import *

# 加载数据
def loadDataset(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		floatArr = map(float, lineArr)
		dataMat.append(floatArr)
	return dataMat

# 计算欧氏距离
def calcEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))

# 随机选择 k 个在边界内的聚类质心点。
# 找出各个维度的max 与 min, 确保质心点的随机初始化值在 各维度的[min, max]之中。
def randCentroids(dataSet, k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k, n)))
	for j in range(n):
		minJ = min(dataSet[:, j])
		maxJ = max(dataSet[:, j])
		rangeJ = float(maxJ - minJ)
		# random.rand(k, 1) 随机生成 k 个 (0, 1) 之间的随机数。
		centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
	return centroids

