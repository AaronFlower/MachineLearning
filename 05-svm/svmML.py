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
def selectLrand(i, m):
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

