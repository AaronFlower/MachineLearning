#coding:utf8

from numpy import *

# 加载数据
def loadDataset(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        row = line.strip().split()
        dataMat.append([float(row[0]), float(row[1])])
        labelMat.append(float(row[2]))
    return dataMat, labelMat

# 从[0,m]中随机取一个数据并且不等于i
def jRand(m, i):
    j = i
    while(i == j):
        j = int(math.floor(random.uniform(0,m)))
    return j

# 计算向量内积。
def kij(dataMat, i, j):
    return dataMat[i,:] * dataMat[j,:].T

# 以新的 alpha 的值进行裁剪
def clipper(alpha, L, H):
    if alpha <= L:
        return L
    if alpha >= H:
        return H
    return alpha


# SMO算法简化版，优化的两个拉格朗日算子不是 huristic 选择的。
def simpleSMO(dataMat, labelMat, C, toler, maxIterNum):
    dataMat = mat(dataMat); labelMat = mat(labelMat).T

    m,n = shape(dataMat)
    alphas = mat(zeros((m, 1))); b = 0
    iterNum = 0 
    while iterNum <= maxIterNum:
        changed = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMat * dataMat[i,:].T)) + b
            Ei = fXi - float(labelMat[i])

            # 在 mEi = labelMat[i] * Ei = labelMat[i] * (Fxi - labelMat[i]) = labelMat[i] * fXi - 1
            # 根据 KKT 条件来判断误差是否真在可以容忍的范围，如果不在则需要优化。
            mEi = labelMat[i] * Ei
            if ((mEi < -toler and alphas[i] < C) or (mEi > toler and alphas[i] > 0)):
	            j = jRand(m, i)
	            fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j,:].T)) + b
	            Ej = fXj - float(labelMat[j])

	            if labelMat[i] == labelMat[j]:
	                aSum = alphas[i] + alphas[j]
	                L = max(0, aSum - C)
	                H = min(C, aSum)
	            else:
	                aDiff = alphas[j] - alphas[i]
	                L = max(0, aDiff)
	                H = min(C, C + aDiff)
	            if L == H : 
	                print 'L == H, continue\n'
	                continue

	            # 计算eta η 
	            eta = 2.0 * kij(dataMat, i, j) - kij(dataMat, i, i) - kij(dataMat, j, j)
	            alphaIold = alphas[i].copy()
	            alphaJold = alphas[j].copy()
	            
	            # eta η 不能为0，大于零没有变化 ？
	            if eta >= 0 : 
	                print 'eta η >= 0 \n'
	                continue
	        
	            alphas[j] -= labelMat[j] * (Ei - Ej) / eta
	            alphas[j] = clipper(alphas[j], L, H)
	            # 如果 alphas[j] 新更新的值前进的 step 不够. not enough
	            if abs(alphas[j] - alphaJold) < 0.00001: 
	                print 'alphas[j] moving up is not enough\n'
	                continue

	            alphas[i] += labelMat[i] * labelMat[j] * (alphaJold - alphas[j])
	            iDiff = alphas[i] - alphaIold; jDiff = alphas[j] - alphaJold
	            b1 = b - iDiff * labelMat[i] * kij(dataMat, i, i) - jDiff * labelMat[j] * kij(dataMat, i, j) - Ei 
	            b2 = b - iDiff * labelMat[i] * kij(dataMat, i, j) - jDiff * labelMat[j] * kij(dataMat,j, j) - Ej 

	            if 0 <= alphas[i] <= C:
	                b = b1
	            elif 0 <= alphas[j] <= C:
	                b = b2
	            else:
	                b = (b1 + b2) / 2

	            changed += 1
        # 内层循环大于 0 ，说明还在继续优化。
        if changed > 0: 
            iterNum = 0
        else:
            iterNum += 1 # 遍历了 maxIterNum 次后，alphas 不再发生变化的进修就可以结束优化了.
        print 'iteration number: %d ' % iterNum
    return b, alphas
