"""
        k近邻算法采⽤测量不同特征值之间的距离⽅法进⾏分类

        本质：提取样本集中特征最相似数据（最近邻）的分类标签,选择样本数据集中前k个最相似的数据

        优点：精度⾼、对异常值不敏感、⽆数据输⼊假定
        缺点：计算复杂度⾼、空间复杂度⾼,需要的存储空间大
        适⽤数据范围：数值型和标称型。
"""

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classifyDistance(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 计算
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 排序
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def fileToMatrix(filename):
    file = open(filename, encoding="utf-8")
    arrayLines = file.readlines()
    lenLines = len(arrayLines)
    returnMat = np.zeros((lenLines, 3))
    classLabelVector = []
    index = 0
    # 解析文件数据
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue - minValue
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minValue, (m, 1))
    # 特征值相除
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minValue


def datingClassTest():
    ratio = 0.1
    datingDataMat, datingDataLabels = fileToMatrix('datingTestSet2.txt')
    _normDataSet, _ranges, _minValue = autoNorm(datingDataMat)
    m = _normDataSet.shape[0]
    testNum = int(m * ratio)
    errorCount = 0
    for i in range(testNum):
        classifierResult = classifyDistance(_normDataSet[i, :], _normDataSet[testNum: m, :],
                                            datingDataLabels[testNum: m], 3)
        print("come back: " + classifierResult + "\t" + "real answer: " + datingDataLabels[i])
        if classifierResult != datingDataLabels[i]:
            errorCount += 1
    print("error ratio: " + str(errorCount / float(testNum)))


def imgToVector(filename):
    returnVector = np.zeros((1, 1024))
    file = open(filename, encoding="utf-8")
    for i in range(32):
        line = file.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(line[j])
    return returnVector


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNunStr = int(fileStr.split('_')[0])
        hwLabels.append(classNunStr)
        trainMat[i, :] = imgToVector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        testFileNameStr = testFileList[i]
        testFileStr = testFileNameStr.split('.')[0]
        testClassNumStr = int(testFileStr.split('_')[0])
        vectorTest = imgToVector('digits/testDigits/%s' % testFileNameStr)
        classifierResult = classifyDistance(vectorTest, trainMat, hwLabels, 3)
        print("comm back: " + str(classifierResult) + "\t" + "real answer: " + str(testClassNumStr))
        if (classifierResult != testClassNumStr): errorCount += 1
    print("error num: " + str(errorCount))
    print("error ratio: " + str(errorCount / float(mTest)))


if __name__ == '__main__':
    # group, labels = createDataSet()
    # distance = classifyDistance([0, 0], group, labels, 3)
    # for i in distance:
    # print(i)
    # dataMat, dataLabels = fileToMatrix("datingTestSet2.txt")
    # print(dataMat)
    # print(dataLabels)
    # figure = plt.figure()
    # ax = figure.add_subplot(111)
    # ax.scatter(dataMat[:, 1], dataMat[:, 2])
    # plt.show()
    # normDataSet, ranges, minValue = autoNorm(dataMat)
    # print(normDataSet)
    # print(ranges)
    # print(minValue)
    # datingClassTest()
    handwritingClassTest()
