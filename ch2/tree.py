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

"""

from math import log


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


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    calcShannonEnt(dataSet)

