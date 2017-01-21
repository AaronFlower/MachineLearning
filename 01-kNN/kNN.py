from numpy import *
import operator

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
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



