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
import operator
import random
import re
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


# 词集模型（每个词的出现与否作为一个特征），构建词向量
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


# 朴素贝叶斯分类器训练函数
# trainCategory 每篇文档中词语的标签构成的向量
# 0 非侮辱 1 侮辱 （每个句子的标签）
# P(c|w) = P(w|c)P(c) / P(w)
def trainNB0(trainMatrix, trainCategory):
    # 行
    numTrainDocs = len(trainMatrix)
    # 列
    numWords = len(trainMatrix[0])
    # 侮辱词语的频率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化概率,根据实际情况来拉普拉斯平滑化
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法
    # 防止遇到的下溢出问题（这是由于太多很小的数相乘造成的），解决方法： 对乘积取自然对数 ln(a*b) = ln(a) + ln(b)
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类器分类函数
# 相乘取对数相加作比较
def classifyNB(vecToClassify, p0Vec, p1Vec, pClass1):
    p1 = sum(vecToClassify * p1Vec) + np.log(pClass1)
    p0 = sum(vecToClassify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 朴素贝叶斯分类器测试函数
def testingNB():
    listOPosts, listClass = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for positionDoc in listOPosts:
        trainMat.append(setOfWordsToVec(myVocabList, positionDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClass))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWordsToVec(myVocabList, testEntry))
    print(str(testEntry) + ' classified as: ' + str(classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWordsToVec(myVocabList, testEntry))
    print(str(testEntry) + ' classified as: ' + str(classifyNB(thisDoc, p0V, p1V, pAb)))


# 词袋模型（一个词在文档中出现不止一次，这能意味着包含该词是否出现在文档中所不能表达的某种信息）
def bagOfWordsToVec(vocabList, inputSet):
    returnVec = len([0] * vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList[word]] += 1
    return returnVec


# 使⽤朴素⻉叶斯对电⼦邮件进⾏分类
# 1. 收集数据：提供⽂本⽂件。
# 2. 准备数据：将⽂本⽂件解析成词条向量
# 3. 分析数据：检查词条确保解析的正确性
# 4. 训练算法：使⽤我们之前建⽴的trainNB()函数
# 5. 测试算法：使⽤classifyNB()，并且构建⼀个新的测试函数来计算⽂档集的错误率。
# 6. 使⽤算法：构建⼀个完整的程序对⼀组⽂档进⾏分类，将错分的⽂档输出到屏幕上。

# 文件解析及完整的垃圾邮件测试函数

def textParse(bigString):
    listOfTokens = re.split(r'\W *', bigString)
    # 筛选少于2字符的字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# ham是不在支持的电子邮件
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 导⼊并解析⽂本⽂件,解析成词列表
        wordList = textParse(open("email/spam/%d.txt" % i, 'r', encoding='utf-8', errors='ignore').read())
        # append和extend都仅只可以接收一个参数，
        # append 任意，甚至是tuple
        # extend 只能是一个列表，其实上面已经说清楚了，是自己没看明白。
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("email/ham/%d.txt" % i, 'r', encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 因为range不能进行del()操作，列表可以进行del()操作。
    trainSet = list(range(50))
    testSet = []
    # 随机构建训练集
    # 留存交叉验证： 随机选择数据的⼀部分作为训练集，⽽剩余部分作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfWordsToVec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    # 计算分类所需的概率
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWordsToVec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("error rate is: " + str(float(errorCount) / len(testSet)))


# 使用朴素贝叶斯来发现地域相关的用词
# 1. 收集数据：从RSS源收集内容，这⾥需要对RSS源构建⼀个接⼝
# 2. 准备数据：将文本文件解析成词条向量
# 3. 分析数据：检查词条确保解析的正确性
# 4. 训练算法：使用我们之前建立的trainNB()函数
# 5. 测试算法：观察错误率，确保分类器可用。可以修改切分程序，以降低错误率，提高分类结果
# 6. 使用算法：构建⼀个完整的程序，封装所有内容。给定两个RSS源， 该程序会显示最常用的公共词

# RSS源分类器及⾼频词去除函数
def calcMostFreq(vocabList, fullText):
    # 计算出现的概率
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    # 倒序排序
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        # 每次访问一条RSS源
        wordList = textParse(feed1['entries'][i]['summary'])


if __name__ == "__main__":
    # listPosts, listClass = loadDataSet()
    # myVocabList = createVocabList(listPosts)+
    # print(myVocabList)
    # # print(setOfWordsToVec(myVocabList, listPosts[0]))
    # trainMat = []
    # for postInDoc in listPosts:
    #     trainMat.append(setOfWordsToVec(myVocabList, postInDoc))
    # print(trainMat)
    # p0V, p1V, pAb = trainNB0(trainMat, listClass)
    # print(p0V)
    # print(p1V)
    # print(pAb)
    # testingNB()
    spamTest()
