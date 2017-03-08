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

# 实现 k-means 算法
# 使用一个 clusterAssignment 矩阵，来存储聚类分配结果矩阵，一列用来记录索引值(聚类类别)，一列存储误差(距离).
def kMeans(dataSet, k, distMeas = calcEclud, createCentroids = randCentroids):
	m = shape(dataSet)[0]
	clusterAssignment = mat(zeros((m, 2)))
	centroids = createCentroids(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		# 为每一个训练通过计算到所聚类的距离，找到一个距离最小的聚类。
		for i in range(m):
			minDist = inf; minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j, :], dataSet[i, :])
				if distJI < minDist :
					minDist = distJI; minIndex = j
			if clusterAssignment[i, 0] != minIndex :
				clusterChanged = True
			clusterAssignment[i, :] = minIndex, minDist ** 2

		# 重新计算 k 个聚类的均值。
		print centroids
		for cent in range(k):
			# 所有为 cent 分类的点, nonzero 为 Indicator 函数。
			pointsInClust = dataSet[nonzero(clusterAssignment[:, 0].A == cent)[0]]
			centroids[cent, :] = mean(pointsInClust, axis = 0)

		# 输出最终分类
		print
		for label in range(k):
			print 'Label %d : %d ' % (label, shape(clusterAssignment[nonzero(clusterAssignment[:, 0] == label)])[1])
			
	return centroids, clusterAssignment
