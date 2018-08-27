# coding:utf-8
from numpy import *

def createDataSet():
	msgList = [
		'I love you',
		'Glad glad glad glad see you',
		'happy happy with you',
		'Sad talk with you sad',
		'I hate hate hate you',
		'I dislike you'
	]
	classList = [0, 0, 0, 1, 1, 1]
	return [msg.split() for msg in msgList], classList


def createVocabList(wordLists):
	vocabSet = set([])
	for wordList in wordLists:
		wdList = [word.lower() for word in wordList]
		vocabSet = vocabSet | set(wdList)
	return list(vocabSet)

# set-of-words model 词集模型
def setOfWords2Vec(vocabList, msg):
	returnVec = [0] * len(vocabList)
	for word in msg:
		if word.lower() in vocabList:
			returnVec[vocabList.index(word.lower())] = 1
	return returnVec

# bag-of-words model 词包模型
def bagOfWords2Vec(vocabList, msg):
	returnVec = [0] * len(vocabList)
	for word in msg:
		if word.lower() in vocabList:
			returnVec[vocabList.index(word.lower())] += 1
	return returnVec

# 获取训练数据
def getTrainingDataSet():
	print ('getTrainingDataSet:')
	msgLists, classList = createDataSet()
	vocabList = createVocabList(msgLists)
	trainingMat = []
	for msg in msgLists:
		#trainingMat.append(setOfWords2Vec(vocabList, msg))
		trainingMat.append(bagOfWords2Vec(vocabList, msg))
	print ('\t', msgLists)
	print ('\t', classList)
	print ('\t', vocabList)
	print ('\t', trainingMat)
	return trainingMat, classList, vocabList

# 训练样本
def trainNaiveBayes(trainingMat, classList):
	numMsgs = len(trainingMat)
	numWords = len(trainingMat[0])
	pDntLike = sum(classList) / float(numMsgs)
	# p0Vec = ones(numWords); p0Denom = 2.0
	# p1Vec = ones(numWords); p1Denom = 2.0	
	p0Vec = zeros(numWords); p0Denom = 0
	p1Vec = zeros(numWords); p1Denom = 0

	for index in range(numMsgs):
		if classList[index] == 0:
			p0Vec += trainingMat[index]
			p0Denom += sum(trainingMat[index])
		else:
			p1Vec += trainingMat[index]
			p1Denom += sum(trainingMat[index])
	#p0Vec = log(p0Vec / p0Denom)
	#p1Vec = log(p1Vec / p1Denom)	
	p0Vec = p0Vec / p0Denom
	p1Vec = p1Vec / p1Denom

	print ('trainNaiveBayes:')
	print ('\t', p0Vec, p1Vec, pDntLike)
	return p0Vec, p1Vec, pDntLike

# Navive Bayes 分类器
def classifyNB0(msgVec, p0Vec, p1Vec, pDntLike):
	p0 = sum(p0Vec * msgVec) + log(1.0 - pDntLike)
	p1 = sum(p1Vec * msgVec) + log(pDntLike)
	if p0 > p1:
		return '0: Like'
	else:
		return '1: dont Like'

# 测试 Naive Bayes 分类
def testNBClassifier():
	trainingMat, labels, vocabList = getTrainingDataSet()
	p0Vec, p1Vec, pDntLike = trainNaiveBayes(trainingMat, labels)
	msg = 'he hate me'
	msgVec = setOfWords2Vec(vocabList, msg.split())
	print (msg, ' calssified as :', classifyNB0(msgVec, p0Vec, p1Vec, pDntLike))
	msg = 'I love you'
	msgVec = setOfWords2Vec(vocabList, msg.split())
	print (msgVec)
	print (msg, ' calssified as :', classifyNB0(msgVec, p0Vec, p1Vec, pDntLike))
	msg = 'hate love like you'
	msgVec = setOfWords2Vec(vocabList, msg.split())
	print (msg, ' calssified as :', classifyNB0(msgVec, p0Vec, p1Vec, pDntLike))
	msg = 'hate dislike love like you'
	msgVec = setOfWords2Vec(vocabList, msg.split())
	print (msg, ' calssified as :', classifyNB0(msgVec, p0Vec, p1Vec, pDntLike))

if __name__ == '__main__':
	testNBClassifier()


