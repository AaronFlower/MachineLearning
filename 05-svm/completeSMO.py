#coding:utf-8

from numpy import *
import math

# 约定与 oS 有关的函数定义，第一个参数传递的都是 optStruct

# 定义一个操作数据结构以方便后继的操作。

class optStruct:
    def __init__(self, dataMat, labelMat, C, toler):
        self.X = dataMat
        self.labelMat = labelMat
        self.C = C
        self.toler = toler
        self.m = shape(dataMat)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 缓存误差。[[1, Ei]] 第一个元素表示误差是否有效，Ei 表示误差。
        self.cacheE = mat(zeros((self.m, 2)))

# 计算 Ek 
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 计算 Kij = K<Xi, Xj>
def Kij(oS, i, j):
    return oS.dataMat[i,:] * os.dataMat[j,:].T

# 从[0, m] 随机选择一个数值 j , 并且 j != i
def selectJrand(i, m):
    j = i 
    while (j == i):
        j = floor(math.uniform(m))
    return j

# 返回[L, H] 之间的 value。
def clipper(value, L, H):
    if value <= L:
        return L
    if value >= H:
        return H
    return value

# 启发时选择可以使步长最大的 alphaJ
def selectJ(oS, i, Ei):
    maxK = -1
    maxDelatE = 0
    Ej = 0
    oS.cacheE[1, Ei]
    #nonzero 会根据传入的数组的维度，返回相应维度的数组。这里第一维是我们非0的行数，正是我们需要的。
    validElist = nonzero(os.cacheE[:, 0])[0] 
    if len(validElist) > 0:
        for k in validElist:
            if k == i : continue
            Ek = calkEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k; maxDeltaE = deltaE; Ej = k
        return maxK, Ej
    else:
        # 初始化时，所有的 Ei 的 valid 属性肯定都是 false ，所以我们还是需要 selectJrand 的。
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 更新 Ek
def update(oS, k):
    Ek = calcEk(oS, k)
    oS.cacheE[k] = [1, Ek]

# Platt SMO 内层优化例程
def innerL(oS, i):
    Ei = calcEk(oS, i);
    sEi = oS.labelMat[i] * Ei
    if ((sEi < -oS.toler) and (oS.alphas[i] < oS.C) or (sEi > oS.toler) and (os.alphas[i] > os.C)):
        j, Ej = selectJ(oS, i, Ei)
        if (oS.labelMat[i] == oS.labelMat[j]):
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        else:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        if L == H:
            print 'L == H'; return 0
        eta = 2.0 * Kij(oS, i, j) - Kij(oS, i, i) - Kij(oS, j, j)
        if eta >= 0: 
            print 'eta >= 0'; return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        os.alphas[j] = clipper(oS.alphas[j], L, H)
        updateEk(oS, j)
        if(abs(oS.alphas[i] - oS.alphas[j]) < 0.00001):
            print 'j not moving enough '; return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - alphaIold)
        updateEk(oS, i)

        diffI = oS.alphas[i] - alphaIold
        diffJ = oS.alphas[j] - alphaJold
        b1 = oS.b - Ei - oS.labelMat[i] * diffI * Kij(oS, i, i) - oS.labelMat[j] * diffJ * Kij(oS, i, j)
        b2 = oS.b - Ej - oS.labelMat[i] * diffI * Kij(oS, i, j) - os.labelMat[j] * diffJ * Kij(oS, j, j)
        if 0 < oS.alphas[i] < oS.C:
            b = b1
        elif 0 < oS.alphas[j] < oS.C:
            b = b2 
        else:
            b = (b1 + b2) / 2.0
        return 1
    else:
        return 0
