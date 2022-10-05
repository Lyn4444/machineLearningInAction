"""
    ⽀持向量机(序列最⼩优化算法)  二分类问题
    ⽀持向量就是离分隔超平⾯最近的那些点

    优点：泛化错误率低，计算开销不⼤，结果易解释
    缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适⽤于处理⼆类问题
    适⽤数据类型：数值型和标称型数据

    原理：
        超平⾯（分类的决策边界），分布在超平⾯⼀侧的所有数据都属于某个类别，⽽分布在另⼀侧的所有数据则属于另⼀个类别。如果数据点离决策边界越远， 那么其最后的预测结果也就越可信。

    松弛变量：
        因为数据不是百分百可分的，要允许有些数据点可以处于分隔⾯的错误⼀侧，这样我们的优化⽬标就能保持仍然不变。

    label * (w^T + b):被称为点到分割面的函数问题

    SVM的⼀般流程：
        1. 收集数据：可以使⽤任意⽅法。
        2. 准备数据：需要数值型数据。
        3. 分析数据：有助于可视化分隔超平⾯。
        4. 训练算法：SVM的⼤部分时间都源⾃训练，该过程主要实现两个参数的调优
        5. 测试算法：⼗分简单的计算过程
        6. 使⽤算法：⼏乎所有分类问题都可以使⽤SVM，值得⼀提的是，SVM本⾝是⼀个⼆类分类器，对多类问题应⽤SVM需要对代码做⼀些修改。

"""
import random

import numpy as np


# SMO: 将⼤优化问题分解为多个⼩优化问题来求解，并且对它们进⾏顺序求解的结果与将它们作为整体来 求解的结果是完全⼀致的
# 简化版SMO算法：简化版会跳过外循环确定要优化的最佳alpha对，在数据集上遍历每⼀个alpha，然后在剩下的alpha集合中随机选择另⼀个alpha，从⽽构建alpha对。
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    file = open(fileName, 'r', encoding='utf-8')
    for line in file.readlines():
        lineArray = line.strip().split('\t')
        # 以数组形式存储
        dataMat.append([float(lineArray[0]), float(lineArray[1])])
        labelMat.append(float(lineArray[2]))
    return dataMat, labelMat


# 随机选择另外一个alpha
def selectOthersAlpha(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(alpha, h, l):
    if alpha > h:
        alpha = h
    if alpha < l:
        alpha = l
    return alpha


# 如果两个向量都不能被优化，退出内循环
# 如果所有向量都没被优化，增加迭代数⽬，继续下⼀次循环
def smoSimple(dataMat, classLabels, C, toleration, maxIter):
    dataMatIn = np.mat(dataMat)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatIn)
    alpha = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            Fi = float(np.multiply(alpha, labelMat).T * (dataMatIn * dataMatIn[i, :].T)) + b
            Ei = Fi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toleration) and (alpha[i] < C)) or (
                    (labelMat[i] * Ei > toleration) and (alpha[i] > 0)):
                # 随机选择的第二个alpha
                j = selectOthersAlpha(i, m)
                Fj = float(np.multiply(alpha, labelMat).T * (dataMatIn * dataMatIn[j, :].T)) + b
                Ej = Fj - float(labelMat[j])
                alphaI = alpha[i].copy()
                alphaJ = alpha[j].copy()
                # 约束条件 C >= a >= 0
                if labelMat[i] != labelMat[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    print("L = H")
                    continue
                eta = 2.0 * dataMatIn[i, :] * dataMatIn[j, :].T - dataMatIn[i, :] * dataMatIn[i, :].T - dataMatIn[j,
                                                                                                        :] * dataMatIn[
                                                                                                             j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alpha[j] -= labelMat[j] * (Ei - Ej) / eta
                alpha[j] = clipAlpha(alpha[j], H, L)
                if abs(alpha[j] - alphaJ) < 0.00001:
                    print("alpha j not moving enough")
                    continue
                alpha[i] += labelMat[j] * labelMat[i] * (alphaJ - alpha[j])
                b1 = b - Ei - labelMat[i] * (alpha[i] - alphaI) * dataMatIn[i, :] * dataMatIn[i, :].T - \
                     labelMat[j] * (alpha[j] - alphaJ) * dataMatIn[i, :] * dataMatIn[j, :].T
                b2 = b - Ej - labelMat[i] * (alpha[i] - alphaI) * dataMatIn[i, :] * dataMatIn[j, :].T - \
                     labelMat[j] * (alpha[j] - alphaJ) * dataMatIn[j, :] * dataMatIn[j, :].T
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed: %d" % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration num: " + str(iter))
    return b, alpha


# Platt SMO算法
# 通过⼀个外循环来选择第⼀个alpha值的，其选择过程会在两种⽅式之间进⾏交替：
#   1.在所有数据集上进⾏单遍扫描
#   2.在⾮边界alpha中实现单遍扫描
# 通过⼀个内循环来选择第⼆个alpha值
# 通过最⼤化步⻓的⽅式来获得第⼆个alpha值，建⽴⼀个全局的缓存⽤于保存误差值，并从中选择使得步⻓或者说Ei-Ej最⼤的alpha值

# 通过建立数据结构来存储缓存过程中的重要值
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toleration):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.toleration = toleration
        self.m = np.shape(dataMatIn)[0]
        self.alpha = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 设置缓存, 第⼀列给出的是Cache是否有效的标志位，⽽第⼆列给出的是实际的E值
        self.cache = np.mat(np.zeros(self.m, 2))


# 计算期望
def calcEk(oS, k):
    Fk = float(np.multiply(oS.alpha, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = Fk - float(oS.labelMat[k])
    return Ek


# 选第二个alpha
def selectJ(i, oS, Ei):
    # 内循环中的启发式⽅法
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.cache[i] = [1, Ei]
    # 矩阵转换成array数组
    valCacheList = np.nonzero(oS.cache[:, 0].A)[0]
    if len(valCacheList) > 1:
        for k in valCacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # 最大步长法
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJ(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 更新缓存
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.cache[k] = [1, Ek]


# Platt SMO算法中的优化例程(寻找决策边界)
def inner(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.toleration and oS.alpha[i] < oS.C) or (
            oS.labelMat[i] * Ei > oS.toleration and oS.alpha[i] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alphaI = oS.alpha[i].copy()
        alphaJ = oS.alpha[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alpha[j] - oS.alpha[i])
            H = min(oS.C, oS.alpha[j] - oS.alpha[i] + oS.C)
        else:
            L = max(0, oS.alpha[j] + oS.alpha[i] - oS.C)
            H = min(oS.C, oS.alpha[j] + oS.alpha[i])
        if L == H:
            print("L = H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alpha[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alpha[j] = clipAlpha(oS.alpha[j], H, L)
        # 更新误差缓存值
        updateEk(oS, j)
        if abs(oS.alpha[j] - alphaJ) < 0.00001:
            print("alpha j not moving enough")
            return 0
        oS.alpha[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJ - oS.alpha[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alpha[i] - alphaI) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                oS.alpha[j] - alphaJ) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alpha[i] - alphaI) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                oS.alpha[j] - alphaJ) * oS.X[j, :] * oS.X[j, :].T
        if 0 < oS.alpha[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alpha[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
            return 1
    else:
        return 0


# 完整的Platt SMO算法外循环代码
def plattSMO(dataMatIn, classLabels, C, toleration, maxIter, kTup=None):
    if kTup is None:
        kTup = {'lin', 0}
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toleration)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            # 遍历所有值
            for i in range(oS.m):
                alphaPairsChanged += inner(i, oS)
                print("fullSet iter: %d    i: %d,  pairs changed  %d " % (iter, i, alphaPairsChanged))
                iter += 1
            else:
                nonBoundIs = np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += inner(i, oS)
                    print("nonBound iter: %d    i: %d,  pairs changed  %d " % (iter, i, alphaPairsChanged))
                    iter += 1
                if entireSet:
                    entireSet = False
                elif alphaPairsChanged == 0:
                    entireSet = True
                print("iteration number: %d" % iter)
            return oS.b, oS.alpha


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(labelArr)
    b, alpha = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alpha[alpha > 0])
    print(np.shape(alpha[alpha > 0]))
    for i in range(100):
        if alpha[i] > 0.0:
            print(dataArr[i], labelArr[i])
