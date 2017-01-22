#coding:utf-8
import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

datingDataSet, labels = kNN.file2matrix('datingTestSet2.txt')

# 获取不同类别的总数，初始化数组.
type1Count = labels.count(1)
type2Count = labels.count(2)
type3Count = labels.count(3)
dataSetType1 = zeros((type1Count, 3))
dataSetType2 = zeros((type2Count, 3))
dataSetType3 = zeros((type3Count, 3))

# 分类
count1 = 0; count2 = 0; count3 = 0; index = 0
for i in labels:
	if i == 1:
		dataSetType1[count1] = datingDataSet[index]
		count1 += 1
	if i == 2:
		dataSetType2[count2] = datingDataSet[index]
		count2 += 1
	if i == 3:
		dataSetType3[count3] = datingDataSet[index]
		count3 += 1
	index += 1

# 'largeDoses':3, 'smallDoses':2, 'didntLike':1
def createPlot0():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	type1 = ax.scatter(dataSetType1[:, 0], dataSetType1[:,1], c='r')
	type2 = ax.scatter(dataSetType2[:, 0], dataSetType2[:,1], c='g')
	type3 = ax.scatter(dataSetType3[:, 0], dataSetType3[:,1], c='b')
	ax.legend([type1, type2, type3], ['didntLike', 'smallDoses', 'largeDoses'])
	plt.xlabel('Flying Miles')
	plt.ylabel('Playing Time')
	plt.show()

def createPlot1():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	type1 = ax.scatter(dataSetType1[:, 0], dataSetType1[:,2], c='r')
	type2 = ax.scatter(dataSetType2[:, 0], dataSetType2[:,2], c='b')
	type3 = ax.scatter(dataSetType3[:, 0], dataSetType3[:,2], c='g')
	ax.legend([type1, type2, type3], ['didntLike', 'smallDoses', 'largeDoses'])
	plt.xlabel('Flying Miles')
	plt.ylabel('Ice-creams')
	plt.show()

def createPlot2():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	type1 = ax.scatter(dataSetType1[:, 1], dataSetType1[:,2], c='r')
	type2 = ax.scatter(dataSetType2[:, 1], dataSetType2[:,2], c='b')
	type3 = ax.scatter(dataSetType3[:, 1], dataSetType3[:,2], c='g')
	ax.legend([type1, type2, type3], ['didntLike', 'smallDoses', 'largeDoses'])
	plt.xlabel('Playing Time')
	plt.ylabel('Ice-creams')
	plt.show()

createPlot0()
createPlot1()
createPlot2()
