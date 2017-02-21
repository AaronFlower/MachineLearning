#coding:utf-8

from numpy import *

# 加载数据
def loadDataset(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        row = line.strip().split('\t')
        dataMat.append([float(row[0]), float(row[1])])
        labelMat.append(float(row[2]))
    return dataMat, labelMat

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
    return oS.X[i,:] * oS.X[j,:].T

# 从[0, m] 随机选择一个数值 j , 并且 j != i
def selectJrand(i, m):
    j = i 
    while (j == i):
        j = int(random.uniform(0,m))
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
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.cacheE[i] = [1, Ei]
    #nonzero 会根据传入的数组的维度，返回相应维度的数组。这里第一维是我们非0的行数，正是我们需要的。
    validElist = nonzero(oS.cacheE[:, 0].A)[0] 
    print 'validList:', validElist, len(validElist)
    if (len(validElist)) > 1:
        for k in validElist:
            if k == i : continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k; maxDeltaE = deltaE; Ej = k
        print 'select:', i, Ei
        print 'if ---', maxK, Ej
        return maxK, Ej
    else:
        # 初始化时，所有的 Ei 的 valid 属性肯定都是 false ，所以我们还是需要 selectJrand 的。
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        print 'select:', i, Ei
        print 'else ---', j, Ej
        return j, Ej

# 更新 Ek
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.cacheE[k] = [1, Ek]

# Platt SMO 内层优化例程
def innerL(oS, i):
    Ei = calcEk(oS, i);
    sEi = oS.labelMat[i] * Ei
    if ((sEi < -oS.toler) and (oS.alphas[i] < oS.C) or (sEi > oS.toler) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(oS, i, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
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
        oS.alphas[j] = clipper(oS.alphas[j], L, H)
        updateEk(oS, j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print 'j not moving enough '; return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - alphaIold)
        updateEk(oS, i)

        diffI = oS.alphas[i] - alphaIold
        diffJ = oS.alphas[j] - alphaJold
        b1 = oS.b - Ei - oS.labelMat[i] * diffI * Kij(oS, i, i) - oS.labelMat[j] * diffJ * Kij(oS, i, j)
        b2 = oS.b - Ej - oS.labelMat[i] * diffI * Kij(oS, i, j) - oS.labelMat[j] * diffJ * Kij(oS, j, j)
        if 0 < oS.alphas[i] < oS.C: oS.b = b1
        elif 0 < oS.alphas[j] < oS.C: oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

# Platt SMO 完整的外层循环代码
def smoP(dataMatIn, labelMatIn, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(labelMatIn).T, C, toler)
    iter = 0
    entireSet = True; alphasPairsChanged = 0
    while (iter < maxIter) and ((alphasPairsChanged > 0) or (entireSet)):
    # while (iter < maxIter):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(oS, i)
            print "fullSet, iter: %d i: %d, paires changed %d " % (iter, i, alphasPairsChanged)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in  nonBoundIs:
                alphasPairsChanged += innerL(i, oS)
                print "non-bound, iter: %d i:%d, parrs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False
        elif (alphasPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
        print 'condition:'
        print iter, alphaPairsChanged, entireSet, maxIter
        print (iter < maxIter) and ((alphasPairsChanged > 0) or (entireSet))
    return oS.b, oS.alphas


