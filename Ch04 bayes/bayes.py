#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/4 21:39
# @Author  : zhcf1ess
# @Site    :
# @File    : bayes.py
# @Software: PyCharm
import numpy as np


##### 过滤网站的恶意留言 #####

def loadDataSet():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 脏话 0 正常话
    return postingList, classVec


def createVocabList(dataSet):
    '''
    获取所有单词的集合
    :param dataSet: 输入数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    '''
    vocabSet = set([])  # 创建空列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符 | 用于求两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''
    遍历查看该单词是否出现 出现该单词则将该单词置1
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    '''
    returnVec = [0] * len(vocabList)  # 创建一个和词汇表等长的向量 将其元素都设置为0
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def _trainNB0(trainMatrix, trainCategory):
    """
    训练数据原版
    :param trainMatrix: 文件单词矩阵 [[1,0,1,1,1....],[],[]...]
    :param trainCategory: 文件对应的类别[0,1,1,0....]，列表长度等于单词矩阵数
                          其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性矩阵
    :return:
    """
    numTrainDocs = len(trainMatrix)  # 文件数
    numWords = len(trainMatrix[0])  # 单词数

    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数
    # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 构造单词出现次数列表
    p0Num = np.zeros(numWords)  # [0,0,0,.....]
    p1Num = np.zeros(numWords)  # [0,0,0,.....]

    # 整个数据集单词出现总数
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        # 是否是侮辱性文件
        if trainCategory[i] == 1:
            # 如果是侮辱性文件，对侮辱性文件的向量进行加和
            p1Num += trainMatrix[i]  # [0,1,1,....] + [0,1,1,....]->[0,2,2,...]
            # 对向量中的所有元素进行求和，也就是计算所有侮辱性文件中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
    # 即 在1类别下，每个单词出现的概率
    p1Vect = p1Num / p1Denom  # [1,2,3,5]/90->[1/90,...]
    # 类别0，即正常文档的[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
    # 即 在0类别下，每个单词出现的概率
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


def trainNB0(trainMatrix, trainCategory):
    """
    训练数据优化版本:
        ·概率为0情况: 向分子分母添加一个非0数
        ·数据下溢: log
    :param trainMatrix: 文件单词矩阵
    :param trainCategory: 文件对应的类别
    :return:
    """
    # 总文件数
    numTrainDocs = len(trainMatrix)
    # 总单词数
    numWords = len(trainMatrix[0])
    # 侮辱性文件的出现概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 构造单词出现次数列表
    # p0Num 正常的统计
    # p1Num 侮辱的统计
    p0Num = np.ones(numWords)  # [0,0......]->[1,1,1,1,1.....]
    p1Num = np.ones(numWords)

    # 整个数据集单词出现总数，2.0根据样本/实际调查结果调整分母的值（2主要是避免分母为0，当然值可以调整）
    # p0Denom 正常的统计
    # p1Denom 侮辱的统计
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 累加辱骂词的频次
            p1Num += trainMatrix[i]
            # 对每篇文章的辱骂的频次 进行统计汇总
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    p1Vect = np.log(p1Num / p1Denom)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    使用算法：
        # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 上面的计算公式，没有除以贝叶斯准则的公式的分母，也就是 P(w)（P(w) 指的是此文档在所有的文档中出现的概率）就进行概率大小的比较了
    # 因为 P(w) 针对的是包含侮辱和非侮辱的全部文档，所以 P(w) 是相同的。
    # 这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)  # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    测试朴素贝叶斯算法
    """
    # 1. 加载数据集
    listOPosts, listClasses = loadDataSet()
    # 2. 创建单词集合
    myVocabList = createVocabList(listOPosts)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 4. 训练数据
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


##### 使用朴素贝叶斯过滤垃圾邮件 #####


def bagOfWords2VecMN(vocabList, inputSet):
    '''
    朴素贝叶斯词袋模型
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return:
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in inputSet:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def setOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):
    '''
    文件解析
    :param bigString: 含大写字母的字符串
    :return: 去掉少于2位的字符串 并且全部转为小写
    '''
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    '''
    对贝叶斯垃圾邮件分类器进行自动化处理。
    :return: 对测试集中的每封邮件进行分类，若邮件分类错误，则错误数加 1，最后返回总的错误百分比。
    '''
    docList = []
    classList = []
    fullText = []

    # 导入文件并解析
    for i in range(1, 26):
        # 切分，解析数据，并归类为 1 类别
        wordList = textParse(open('spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        # 切分，解析数据，并归类为 0 类别
        wordList = textParse(open('ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 创建词汇表
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []

    # 随机取 10 个邮件用来测试
    for i in range(10):
        # random.uniform(x, y) 随机生成一个范围为 x - y 的实数
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    # k折交叉验证
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0

    # 测试训练集
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('the errorCount is: ', errorCount)
    print('the testSet length is :', len(testSet))
    print('the error rate is :', float(errorCount) / len(testSet))


##### 使用朴素贝叶斯分类器从个人广告中获取区域倾向 #####

def calcMostFreq(vocabList, fullText):
    '''
    RSS源分类器及高频词去除函数
    :param vocabList:
    :param fullText:
    :return:
    '''
    import operator
    freqDict = {}
    for token in vocabList:  # 遍历词汇表中的每个词
        freqDict[token] = fullText.count(token)  # 统计每个词在文本中出现的次数
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)  # 根据每个词出现的次数从高到底对字典进行排序
    return sortedFreq[:30]  # 返回出现次数最高的30个单词


def localWords(feed1, feed0):
    '''

    :param feed1: RSS源1
    :param feed0: RSS源2
    :return:
    '''
    import feedparser
    docList = []
    classList = []
    fullText = []
    minlen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minlen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 两个RSS源作为正反例
    vocabulary = createVocabList(docList)
    # 创建词汇库
    top30Words = calcMostFreq(vocabulary, fullText)
    # 获得出现频率最高的30个
    for pairW in top30Words:
        if pairW[0] in vocabulary: vocabulary.remove(pairW[0])
    # 去除前30的单词

    trainingSet = list(range(2 * minlen))
    # print(trainingSet)
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        print(trainingSet[randIndex])
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    # 随机选择训练和测试集；测试集为20个
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabulary, docList[docIndex]))
        trainClass.append(classList[docIndex])
    # 将训练集内的文档转换成频数特征
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabulary, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    # import feedparser
    # docList = []
    # classList = []
    # fullText = []
    # minlen = min(len(feed1['entries']), len(feed0['entries']))
    # for i in range(minlen):
    #     wordList = textParse(feed1['entries'][i]['summary'])
    #     docList.append(wordList)
    #     fullText.extend(wordList)
    #     classList.append(1)
    #     wordList = textParse(feed0['entries'][i]['summary'])
    #     docList.append(wordList)
    #     fullText.extend(wordList)
    #     classList.append(0)
    # # 两个RSS源作为正反例
    # vocabulary = createVocabList(docList)
    # # 创建词汇库
    # top30Words = calcMostFreq(vocabulary, fullText)
    # # 获得出现频率最高的30个
    # for pairW in top30Words:
    #     if pairW[0] in vocabulary: vocabulary.remove(pairW[0])
    # # 去除前30的单词
    #
    # trainingSet = list(range(2 * minlen))
    # testSet = []
    # for i in range(20):
    #     randIndex = int(np.random.uniform(0, len(trainingSet)))
    #     testSet.append(trainingSet[randIndex])
    #     del (trainingSet[randIndex])
    # # 随机选择训练和测试集；测试集为20个
    # trainMat = []
    # trainClass = []
    # for docIndex in trainingSet:
    #     trainMat.append(bagOfWords2VecMN(vocabulary, docList[docIndex]))
    #     trainClass.append(classList[docIndex])
    # # 将训练集内的文档转换成频数特征
    # p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    # errorCount = 0
    # for docIndex in testSet:
    #     wordVector = bagOfWords2VecMN(vocabulary, docList[docIndex])
    #     if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
    #         errorCount += 1
    # print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabulary, p0V, p1V


# def stopWords():
#     import re
#     wordList = open('stopword.txt').read()  # see http://www.ranks.nl/stopwords
#     listOfTokens = re.split(r'\W*', wordList)
#     return [tok.lower() for tok in listOfTokens]
#     print('read stop word from \'stopword.txt\':', listOfTokens)
#     return listOfTokens


def getTopWords(ny, sf):
    '''
    最具表征性的词汇显示函数
    :param ny: RSS源
    :param sf: RSS源
    :return:
    '''
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)  # 用贝叶斯分类器训练并测试

    # 用于元组储存
    topNY = []
    topSF = []

    # 返回大于阈值的值 并排序
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')

    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


'''
朴素贝叶斯工作原理:

    ·提取所有文档中的词条并进行去重
    ·获取文档的所有类别
    ·计算每个类别中的文档数目
    ·对每篇训练文档: 
        对每个类别: 
            如果词条出现在文档中-->增加该词条的计数值（for循环或者矩阵相加）
            增加所有词条的计数值（此类别下词条总数）
    ·对每个类别: 
        对每个词条: 
            将该词条的数目除以总词条数目得到的条件概率（P(词条|类别)）
    ·返回该文档属于每个类别的条件概率（P(类别|文档的所有词条)）
'''

'''
朴素贝叶斯开发过程:

    ·收集数据: 可以使用任何方法。
    ·准备数据: 需要数值型或者布尔型数据。
    ·分析数据: 有大量特征时，绘制特征作用不大，此时使用直方图效果更好。
    ·训练算法: 计算不同的独立特征的条件概率。
    ·测试算法: 计算错误率。
    ·使用算法: 一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定非要是文本。
'''

'''
朴素贝叶斯 算法特点:

    ·优点: 在数据较少的情况下仍然有效，可以处理多类别问题。
    ·缺点: 对于输入数据的准备方式较为敏感。
    ·适用数据类型: 标称型数据。
'''
import feedparser

ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')

# print(ny)
# print(sf)

vocabulary, pSF, pNY = localWords(ny, sf)
