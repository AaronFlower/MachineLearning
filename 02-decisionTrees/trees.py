#coding:utf-8

from math import log
import operator

def createDataSet ():
	# dataSet 不可以直接用 numpy.array 来存储，因为 numpy.array 要求数组元素的类型要一致。
	dataSet = [
		[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']
	]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

# 计算熵。 熵越高，则混合的数据也越多。
def calcShannonEnt (dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1

	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

# 划分数据集。
def splitDataSet (dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet: 
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis + 1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

# 找出最好划分数据集的特征值
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	baseInfoGain = 0; bestFeature = -1
	for i in range(numFeatures):
		featureList = [example[i] for example in dataSet]
		uniqueVals = set(featureList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)		
		infoGain = baseEntropy - newEntropy
		print '%d, baseEntropy: %f, newEntropy : %f , infoGain: %f' % (i, baseEntropy, newEntropy, infoGain)
		if (infoGain > baseInfoGain):
			baseInfoGain = infoGain
			bestFeature = i
	return bestFeature

# 当所有特征值都用完的时候，用多数表决的方法来决定分类。
def majorityCnt (classList):
	classCount = {}
	for vote in classList:
		classCount[vote] = classCount.get(vote, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

# 递归创建决策树
def createTree(dataSet, labels):
	# 递归函数开始最重要的事情结束条件判断
	# 1. 类别完全相同则停止继续划分
	print '---->>>>>>>>'
	print dataSet
	print labels
	print '<<<<------'
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 2. 或当没有特征值可以继续用于划分的时候，则停止，直接返回多数表决。
	if len(dataSet[0]) == 1:
		return majorityCnt(dataSet)

	bestFeature = chooseBestFeatureToSplit(dataSet)
	bestFeatureLabel = labels[bestFeature]
	myTree = {bestFeatureLabel: {}}
	del(labels[bestFeature])
	featureVals = [example[bestFeature] for example in dataSet]
	uniqueVals = set(featureVals)
	for val in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatureLabel][val] = createTree(splitDataSet(dataSet, bestFeature, val), subLabels)

	return myTree