#coding:utf-8
from numpy import *
def loadDataSet():
	postingList = [
		['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
		['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
		['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
		['stop', 'posting', 'stupid', 'worthless', 'garbage'],
		['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
		['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
	]
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec

# 创建 word vocabulary 
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

# 将评论转换成单词本对应的向量。 词集模型, set-of-words model
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print ('the word: %s is not in my Vocabulary!' % word)
	return returnVec

# 词包模型, bag-of-words model
def bagOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

# 获取训练样本
def getTrainingDataSet():
	postingList, trainCategory = loadDataSet()
	wordVocab = createVocabList(postingList)
	trainMatrix = []
	for postList in postingList:
		trainMatrix.append(bagOfWords2Vec(wordVocab, postList))
	return trainMatrix, trainCategory

def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = ones(numWords); p0Denom = 2.0 # 初始化单词数个数为1， 避免概率为0时个影响。
	p1Num = ones(numWords); p1Denom = 2.0	
	# p0Num = zeros(numWords); p0Denom = 0.0 
	# p1Num = zeros(numWords); p1Denom = 0.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1: 
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	# p1Vect = p1Num / p1Denom
	# p0Vect = p0Num / p0Denom	
	p1Vect = log(p1Num / p1Denom)
	p0Vect = log(p0Num / p0Denom)
	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	print ('--->>>>')
	print ('p0 %f, p1 %f' % (p0, p1))
	print ('<<<<---')
	if p1 > p0:
		return 1
	else :
		return 0

# 测试 Naive Bayes 分类.
def testingNB():
	dataSet, trainCategory = loadDataSet()
	myVocabList = createVocabList(dataSet)
	trainMatrix, trainCategory = getTrainingDataSet()
	p0V, p1V, pAb = trainNB0(trainMatrix, trainCategory)
	testEntry = ['love', 'love', 'my', 'dalmation']
	thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
	result = classifyNB(thisDoc, p0V, p1V, pAb)
	print (testEntry, 'classified as:', 	result)
	testEntry = ['stupid', 'garbage', 'stupid']
	thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
	result = classifyNB(thisDoc, p0V, p1V, pAb)
	print (testEntry, 'classified as:', 	result)

# Ham, A message is called ham when it's not spam.
# 文本解析及完整的垃圾邮件测试函数.

def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []; classList = []; fullText = []
	for i in range(1, 26):
		wordList = textParse(open('Ch04/email/spam/%d.txt' % i, errors='ignore', encoding='utf-8').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('Ch04/email/ham/%d.txt' % i, errors='ignore', encoding='utf-8').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)

	vocabList = createVocabList(docList)
	trainingSet = list(range(50)); testSet = []
	# 随机选择 40 个训练样本 和 10 个测试数据。
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])

	trainingMat = []
	trainingClasses = []
	for docIndex in trainingSet:
		trainingMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainingClasses.append(classList[docIndex])

	p0V, p1V, pSpam = trainNB0(array(trainingMat), array(trainingClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		classResult = classifyNB(array(wordVector), p0V, p1V, pSpam)
		print ('docIndex %d: came back is : %s , the real class is: %s' % (docIndex, classResult, classList[docIndex]))
		if classResult != classList[docIndex]:
			errorCount += 1
	print ('The error rate is:', float(errorCount) / len(testSet))

# if __name__ == '__main__':
spamTest()