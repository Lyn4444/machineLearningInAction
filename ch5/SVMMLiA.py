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


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(labelArr)
    b, alpha = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alpha[alpha > 0])
