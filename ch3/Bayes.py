"""
    基于概率论的分类⽅法：朴素⻉叶斯

    将⽂档切分成词向量，然后利⽤词向量对⽂档进⾏分类

    朴素贝叶斯

        优点：在数据较少的情况下仍然有效，可以处理多类别问题。
        缺点：对于输⼊数据的准备⽅式较为敏感。
        适⽤数据类型：标称型数据。

    条件概率：
        P(c|x) = P(x|c)P(c) / P(x)

    独⽴:
        统计意义上的独⽴，即⼀个特征或者单词出现的可能性与它和其他单词相邻没有关系

    朴素⻉叶斯的⼀般过程:
        1. 收集数据：可以使⽤任何⽅法。
        2. 准备数据：需要数值型或者布尔型数据
        3. 分析数据：有⼤量特征时，绘制特征作⽤不⼤，此时使⽤直⽅图效果更好。
        4. 训练算法：计算不同独⽴特征的条件概率。
        5. 测试算法：计算错误率。
        6. 使⽤算法：⼀个常⻅的朴素⻉叶斯应⽤是⽂档分类。可以在任意的分类场景中使⽤朴素⻉叶斯分类器，不⼀定⾮要是⽂本。
"""
import numpy as np


def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 创建⼀个包含在所有⽂档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWordsToVec(vocabList, inputSet):
    # 创建一个其中含有元素为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历⽂档中的所有单词，如果出现了词汇表中的单词，则将输出的⽂档向量中的对应值设为1。
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(word + "----- not in vocabulary")
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numWords)
    # 初始化概率
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)



if __name__ == "__main__":
    listPosts, listClass = loadDataSet()
    myVocabList = createVocabList(listPosts)
    # print(myVocabList)
    # print(setOfWordsToVec(myVocabList, listPosts[0]))
