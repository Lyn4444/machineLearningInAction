"""
    Logistic回归(最优化算法)
        优点：计算代价不⾼，易于理解和实现。
        缺点：容易⽋拟合，分类精度可能不⾼。
        适⽤数据类型：数值型和标称型数据。

    回归：
        假设现在有⼀些数据点，我们⽤⼀条直线对这些点进⾏拟合（该线称为最佳拟合直线），这个拟合过程就称作回归

    Logistic回归的⼀般过程：
        1. 收集数据：采⽤任意⽅法收集数据。
        2. 准备数据：由于需要进⾏距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
        3. 分析数据：采⽤任意⽅法对数据进⾏分析。
        4. 训练算法：⼤部分时间将⽤于训练，训练的⽬的是为了找到最佳的分类回归系数
        5. 测试算法：⼀旦训练步骤完成，分类将会很快
        6. 使⽤算法：⾸先，我们需要⼀些输⼊数据，并将其转换成对应的结构化数值；接着，基于训练好的回归系数就可以对这些数值进⾏简单的回归计算，判定它们属于哪个类别；在这之后，我们就可以在输出的类别上做⼀些其他分析⼯作。

"""
import math
import random

import matplotlib.pyplot as plt
import numpy as np


# 基于最优化⽅法的最佳回归系数确定
# （1）梯度上升法：要找到某函数的最⼤值，最好的⽅法是沿着该函数的梯度⽅向探寻，梯度算⼦总是指向函数值增⻓最快的⽅向。这⾥所说的是移动⽅向，⽽未提到移动量的⼤⼩。该量值称为步⻓，记做alpha
# 梯度上升算法⽤来求函数的最⼤值，⽽梯度下降算法⽤来求函数的最⼩值。
def loadDataSet():
    dataMat = []
    labelMat = []
    file = open('textSet.txt', 'r', encoding='utf-8')
    for line in file.readlines():
        lineArray = line.strip().split()
        dataMat.append([1.0, float(lineArray[0]), float(lineArray[1])])
        labelMat.append(int(lineArray[2]))
    return dataMat, labelMat


# Sigmoid函数（阶越函数）
# f(Z) = 1 / (1 + e^(-z))
def sigmoid(inX):
    #  #对sigmoid函数的优化，避免了出现极大的数据溢出
    if inX >= 0:
        return 1.0 / (1 + np.exp(-inX))
    else:
        return np.exp(inX) / (1 + np.exp(inX))


# 梯度上升法
# sigmoid函数的输入z：z = W^T * x(W^T为回归系数矩阵的转置)
# W = W + alpha * f(W)的梯度
def gradAscend(dataMatIn, classLabels):
    # 转换为NumPy矩阵数据类型
    dataMat = np.mat(dataMatIn)
    # transpose： 矩阵转置
    labelsMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMat)
    # 步长
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 回归系数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = (labelsMat - h)
        weights = weights + alpha * dataMat.transpose() * error
    return weights


# 画出数据集和logistic回归最佳拟合直线的函数
def plotBestFit(weight):
    # weights = weight.getA()
    weights = weight
    dataMat, labelMat = loadDataSet()
    dataArray = np.array(dataMat)
    n = np.shape(dataArray)[0]
    xCord1 = []
    yCord1 = []
    xCord2 = []
    yCord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xCord1.append(dataArray[i, 1])
            yCord1.append(dataArray[i, 2])
        else:
            xCord2.append(dataArray[i, 1])
            yCord2.append(dataArray[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xCord1, yCord1, s=30, c='red', marker='s')
    ax.scatter(xCord2, yCord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # 最佳拟合直线
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法；一次仅用一个样本点来更新回归系数
# 区别：
#   1.梯度上升算法的变量h和误差error都是向量,随机梯度上升算法则全是数值型
#   2.随机梯度上升算法没有矩阵的转换过程，所有变量的数据类型都是NumPy数组
def stoGradAscent0(dataMat, classLabels):
    m, n = np.shape(dataMat)
    alpha = 0.01
    weight = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(dataMat[i] * weight))
        error = classLabels[i] - h
        weight = weight + alpha * error * dataMat[i]
    return weight


# 随机梯度上升算法的回归系数经过⼤量迭代才能达到稳定值，并且仍然有局部的波动现象
# 原因：存在⼀些不能正确分类的样本点（数据集并⾮线性可分），在每次迭代时会引发系数的剧烈改变。我们期望算法能避免来回波动，从⽽收敛到某个值,同时收敛速度也需要加快。
# 改进的随机梯度上升算法
def stoGradAscent1(dataMat, classLabels, numIter=50):
    m, n = np.shape(dataMat)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha每次迭代时都需要调整来缓解波动，常数项是为了保证在多次迭代之后新数据依旧有一定的影响力，如果处理的是动态数据则需要加大常数项来确保获得更大的回归系数
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选取更新
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del (dataIndex[randIndex])
    return weights


# Logistic回归估计病马的死亡率
# 处理数据中的缺失值
#   1.使用可用特征的均值来填补缺失值
#   2.使用特殊值来填补，如-1
#   3.忽略有缺失值的样本
#   4.使用相似的样本均值来填补
#   5.使用另外的机器学习算法来预测缺失值

# 如果是数据的类别标签缺失则会把该数据丢弃
# 以回归系数和特征向量作为输⼊来计算对应的Sigmoid值
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    fileTrain = open('horseColicTraining.txt')
    fileTest = open('horseColicTest.txt')
    trainSet = []
    trainLabel = []
    for line in fileTrain.readlines():
        _line = line.strip().split('\t')
        lineArray = []
        for i in range(21):
            lineArray.append(float(_line[i]))
        trainSet.append(lineArray)
        trainLabel.append(float(_line[21]))
    trainWeghts = stoGradAscent1(np.array(trainSet), trainLabel, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in fileTest.readlines():
        numTestVec += 1.0
        _line = line.strip().split("\t")
        lineArray = []
        for i in range(21):
            lineArray.append(float(_line[i]))
        if int(classifyVector(np.array(lineArray), trainWeghts)) != int(_line[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("error rate is: " + str(errorRate))
    return errorRate


def multiTest():
    numTest = 10
    errorNum = 0.0
    for k in range(numTest):
        errorNum += colicTest()
    print(str(numTest) + " iterations the rate is: " + str(errorNum / float(numTest)))


if __name__ == "__main__":
    dataArr, labelsMat = loadDataSet()
    # weight = gradAscend(dataArr, labelsMat)
    # print(dataArr)
    # print(labelsMat)
    # getA(): 将矩阵转成数组的形式
    # print(weight.getA())
    # plotBestFit(weight.getA())
    # weight = stoGradAscent1(np.array(dataArr), labelsMat, 5000)
    # print(weight)
    # plotBestFit(weight)
    multiTest()
