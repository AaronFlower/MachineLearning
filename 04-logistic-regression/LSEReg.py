import numpy as np 
import matplotlib.pyplot as plt 

def loadData(file):
  f = open(file, 'r')
  features = []
  labels = []
  for line in f:
      segs = line.strip().split()
      features.append([1.0, float(segs[0]), float(segs[1])])
      labels.append([int(segs[2])])
  return features, labels

def LSE_weights(features, labels):
	''' 最小二乘法计算出 weights '''
	X = np.mat(features)
	y = np.mat(labels)
	return (X.T * X).I * X.T * y

def plotRegression(file):
  features, labels = loadData(file)
  weights = LSE_weights(features, labels)
  classes = np.unique(labels)
  fArr = np.array(features)
  lArr = np.array(labels)
  negativeMask = np.squeeze(lArr == classes[0])
  positiveMask = np.squeeze(lArr == classes[1])
  negativeSamples = fArr[negativeMask] # fancy indexing
  positiveSamples = fArr[positiveMask]
  plt.scatter(negativeSamples[:, 1], negativeSamples[:, 2], marker='x')
  plt.scatter(positiveSamples[:, 1], positiveSamples[:, 2], marker='o')
  x_1 = np.arange(-4, 4, 0.1)
  x_2 = -(weights[0] + weights[1] * x_1) / weights[2] 
  plt.plot(x_1, np.squeeze(np.asarray(x_2)))
  plt.show()

def main():
	plotRegression('Ch05/testSet01.txt')
	plotRegression('Ch05/testSet02.txt')

if __name__ == '__main__':
	main()
