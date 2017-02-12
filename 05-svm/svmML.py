# coding: utf8
# SVM

import math
from numpy import *

# load data 加载数据。
def loadDataSet(fileName):
  dataMat = []; labelMat = []
  fr = open(fileName)
  for line in fr.readlines():
    lineArr = line.strip().split('\t')
    dataMat.append([float(lineArr[0]), float(lineArr[1])])
    labelMat.append(float(lineArr[2]))
  return dataMat, labelMat

# 从（0, m） 中随机选择一个数值 j ,并且使得 j != i
def selectJrand(i, m):
  j = i
  while( j == i):
      j = int(random.uniform(0, m))
  return j

# 裁剪 aj 使得 H >= aj >= L
def clipAlpha(aj, H, L):
  if aj > H:
      aj = H
  if L > aj:
      aj = L
  return aj

# 简化版 SMO 算法
def smoSimple(dataMatIn, classLabels, C, toler, matIter):
    dataMat = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0
    m,n = shape(dataMat)
    alphas = mat(zeros((m, 1)))
    iterNum = 0
    while (iterNum < matIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler and alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j, : ].T )) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = max(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = max(C, alphas[j] + alphas[i])
                if L == H:
                    print "L == H"; continue

                sumOfIJ = dataMat[i,:] * dataMat[i,:].T + dataMat[j,:] * dataMat[j, :].T 
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - sumOfIJ 
                
                if eta >= 0:
                    print "eta >= 0"; continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "J not moving enough"; continue

                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMat[i, :] * dataMat[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[j, :].T - labelMat[j] *(alphas[j] - alphaJold) * dataMat[j, :] * dataMat[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]): 
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iterNum, i, alphaPairsChanged)
            
            if (alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0

            print "iteration number: %d" % iterNum
        return b, alphas
