import numpy as np 
import matplotlib.pyplot as plt 

def loadData():
	''' 测试数据有 100 个样本，样本中有两个特性，标记分类为 0, 1 '''
	file = open('testSet.txt', 'r')
	features = []
	labels = []
	for line in file:
		segs = line.split()
		features.append([1, float(segs[0]), float(segs[1])])
		labels.append([int(segs[2])])

	return features, labels

def LSE_weights(X, y):
	''' 最小二乘法计算出 weights '''
	return (X.T * X).I * X.T * y

def plotRegression(X, y, weights):
	''' 绘制出回归的分类图 '''
	positives = np.array([sample for inx, sample in enumerate(X) if y[inx][0] == 1 ])
	negatives = np.array([sample for inx, sample in enumerate(X) if y[inx][0] == 0 ])
	plt.scatter(positives[:, 1], positives[:, 2])
	plt.scatter(negatives[:, 1], negatives[:, 2], marker='x')
	plt.show()

def main():
	features, labels = loadData()
	weights = LSE_weights(np.mat(features), np.mat(labels))
	plotRegression(features, labels, weights)

if __name__ == '__main__':
	main()
