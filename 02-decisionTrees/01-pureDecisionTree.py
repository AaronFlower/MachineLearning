'''
encoding: utf-8
'''
from numpy import *
from math import log
import operator

'''
计算香农熵 Shannon Entropy
统计各个符号出现的概率，使用香农熵公式来计算即可。
'''
def calcShannonEntropy(dataset):
	labels = [example[-1] for example in dataset]
	entries = len(labels)
	labelsSet = set(labels)
	entropy = 0.0
	for label in labelsSet:
		prob = labels.count(label) / entries
		entropy += - prob * log(prob, 2)
	return entropy

'''
根据相应的特征及特征的取值，获取对的子数据集。
'''
def getSplitDataset(dataset, featureIndex, value):
	subDataset = []
	for example in dataset:
		if example[featureIndex] == value:
			subDataset.append(example[:featureIndex] + example[featureIndex + 1 :])
	return subDataset

'''
根据信息增益来获取最优的划分数据集的特征
'''
def chooseBestSplitFeature(dataset, labels):
	numLabels = len(labels)
	entries = len(dataset)
	baseEntropy = calcShannonEntropy(dataset)
	bestFeature = -1; bestInfoGain = 0
	for i in range(numLabels):
		values = [example[i] for example in dataset]
		valueSet = set(values)
		entropy = 0.0
		for value in valueSet:
			subDataset = getSplitDataset(dataset, i, value)
			subEntropy = calcShannonEntropy(subDataset)
			entropy += len(subDataset) / entries * subEntropy

		infoGain = baseEntropy - entropy
		if infoGain >= bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

'''
子树投票，当特征可以使用的时候需要用投票的方式来决定子树的分类。
'''
def getMajorityClass(dataset):
	labelsCount = {}
	for example in dataset:
		labelsCount[example[-1]] = labelsCount.get(example[-1], 0) + 1
	sortedLabels = sorted(labelsCount.items(), key = operator.itemgetter(1), reverse = True)
	return sortedLabels[0][0]

'''
递归创建子树
'''
def createDecisionTree(dataset, features):
	labels = [example[-1] for example in dataset]
	# 数据集已经分类无需再分。
	if (len(labels) == labels.count(labels[0])):
		return labels[0]
	# 没有特征可以使用了。使用投票方式来划分。 
	if (len(features) == 0):
		return getMajorityClass(dataset)
	bestFeatureIndex = chooseBestSplitFeature(dataset, features)
	featureLabel = features[bestFeatureIndex]
	tree = {featureLabel: {}}
	del features[bestFeatureIndex]
	values = [example[bestFeatureIndex] for example in dataset]
	valuesSet = set(values)
	for value in valuesSet:
		subFeatures = features[:]
		tree[featureLabel][value] = createDecisionTree(getSplitDataset(dataset, bestFeatureIndex, value), features)

	return tree

'''
创建简单数据集
'''
def createDataSet():
	dataset = [
		[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']
	]
	labels = ['no surfacing', 'flippers']
	return dataset, labels

if __name__ == '__main__':
	print('Hello python main, all about decision tree!')
	dataset, features = createDataSet()
	print('\t', dataset, features)
	print('Base Shannon Entropy')
	print('\t', calcShannonEntropy(dataset))
	print('Get sub data set')
	print('\t', getSplitDataset(dataset, 0, 1))
	print('\t', getSplitDataset(dataset, 0, 0))
	print('\t', getSplitDataset(dataset, 1, 1))
	print('\t', getSplitDataset(dataset, 1, 0))
	print('Choose best feature to split')
	print('\t', chooseBestSplitFeature(dataset, features))
	print('Sorted labels')
	print('\t', getMajorityClass(dataset))
	print('Create Decision tree')
	print('\t', createDecisionTree(dataset, features))
	print("expect result: ")
	print("\t {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}")
