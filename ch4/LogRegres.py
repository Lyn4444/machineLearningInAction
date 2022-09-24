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
    return 1.0 / (1 + np.exp(-inX))


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
    # getA(): 将矩阵转成数组的形式
    weights = weight.getA()
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


if __name__ == "__main__":
    dataArr, labelsMat = loadDataSet()
    weight = gradAscend(dataArr, labelsMat)
    # print(dataArr)
    # print(labelsMat)
    print(weight.getA())
    plotBestFit(weight)
