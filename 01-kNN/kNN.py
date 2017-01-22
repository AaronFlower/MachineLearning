#coding:utf-8
from numpy import *
import operator

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def createFilmDataSet(): 
	group = array([
		[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]
	])
	labels = ['love', 'love', 'love', 'action', 'action', 'action']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dateSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dateSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDiffMatDistance = sqDiffMat.sum(axis = 1)
	distances = sqDiffMatDistance ** 0.5
	distancesSortedIndicies = distances.argsort()

	classCount = {}
	for i in range(k):
		voteLabel = labels[distancesSortedIndicies[i]]
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0 : 3]
		if(listFromLine[-1].isdigit()):
		    classLabelVector.append(int(listFromLine[-1]))
		else:
		    classLabelVector.append(love_dictionary.get(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

# normalizing values
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet * 1.0 / tile(ranges, (m, 1)) 
	return normDataSet, ranges, minVals

# 假设用 90% 的数据作为训练样本，用 10% 的数据作为测试数据。并统计出错误率。
def datingClassTest ():
	hoRatio = 0.1
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0
	for i in range(numTestVecs):
		result = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
		print "The classifier came back with: %d, the real answer is: %d " % (result, datingLabels[i])
		if (result != datingLabels[i]):
			errorCount += 1.0
	print 'The total error rate is: %f' % (errorCount/float(numTestVecs))

def classifyPerson ():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(raw_input('percentage of time spend playing video games?'))
	ffMiles = float(raw_input('frequent filier miles earned per year?'))
	iceCream = float(raw_input('liters of ice cream consumed per year?'))

	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	result = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)

	print "You will probably like this person:", resultList[result - 1]
