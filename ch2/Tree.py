"""
    决策树

    优点：计算复杂度不⾼，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。
    缺点：可能会产⽣过度匹配问题。

    适⽤数据类型：数值型和标称型。

    决策树的⼀般流程

    1. 收集数据：可以使⽤任何⽅法。
    2. 准备数据：树构造算法只适⽤于标称型数据，因此数值型数据必须离散化。
    3. 分析数据：可以使⽤任何⽅法，构造树完成之后，我们应该检查图形 是否符合预期。
    4. 训练算法：构造树的数据结构。
    5. 测试算法：使⽤经验树计算错误率。
    6. 使⽤算法：此步骤可以适⽤于任何监督学习算法，⽽使⽤决策树可以 更好地理解数据的内在含义。

    信息增益

    划分数据集的⼤原则是：将⽆序的数据变得更加有序。

    对于过度划分的数据集，可以删掉部分不重要信息节点，如CART/ID3算法

"""

from math import log
import operator
import TreePlotter
import pickle


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        # 以2为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            # 计算最好的信息熵
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 三个输⼊参数：待划分的数据集、划分数据集的 特征、需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
            # 利用operator操作键值进行字典排序
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历返回出现最多次数的特征值
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    newTree = {bestFeatLabel: {}}
    # 得到列表中包含的属性值
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        newTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return newTree


def classify(myTree, featLabels, testVec):
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        # 将标签转便成索引
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 使用pickle模块存储决策树
def storeTree(myTree, filename):
    file = open(filename, 'wb')
    pickle.dump(myTree, file)
    file.close()


def grabTree(filename):
    file = open(filename, 'rb')
    return pickle.load(file)


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    print(dataSet)
    # print(labels)
    # resDataSet = splitDataSet(dataSet, 0, 0)
    # print(resDataSet)
    # shannonEnt = calcShannonEnt(dataSet)
    # print(shannonEnt)
    # chooseDataSet = chooseBestFeatureToSplit(dataSet)
    # print(chooseDataSet)
    # myTree = createTree(dataSet, labels)
    # print(myTree)
    # myData, label = createDataSet()
    # print(label)
    # myTree = treePlotter.retrieveTree(0)
    # print(classify(myTree, label, [1, 0]))
    # print(classify(myTree, label, [1, 1]))
    # storeTree(myTree, 'classifierStorage.txt')
    # print(grabTree('classifierStorage.txt'))
    # file = open("lenses.txt")
    # lenses = [line.strip().split('\t') for line in file.readlines()]
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # lensesTree = createTree(lenses, lensesLabels)
    # print(lensesTree)
    # TreePlotter.createPlot(lensesTree)
