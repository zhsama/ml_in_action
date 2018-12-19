#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/16 8:55
# @Author  : zhcf1ess
# @Site    : 
# @File    : adaboost.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树分类函数
    Args:
        dataMatrix: 数据矩阵
        dimen: 第dimen列，也就是第几个特征
        threshVal: 阈值
        threshIneq: 标志

    Returns:
        retArray: 分类结果

    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    找到数据集上最佳的单层决策树
    Args:
        dataArr: 数据矩阵
        classLabels: 数据标签
        D: 最佳分类结果

    Returns:
        bestStump: 最佳单层决策树信息
		minError: 最小误差
		bestClasEst: 最佳的分类结果

    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  # 初始化错误数为正无穷,便于寻找最小错误率
    # 遍历数据集的所有特征
    for i in range(n):  # 遍历数据的所有维度
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 在当前维度的所有范围内循环
        for j in range(-1, int(numSteps) + 1):
            # 在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 让树桩以i, j, lessThan变量分类
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # 计算加权错误率(与分类器进行交互)
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    使用AdaBoost算法训练分类器
    Args:
        dataArr: 数据矩阵
        classLabels: 数据标签
        numIt: 最大迭代次数

    Returns:
        weakClassArr: 训练好的分类器
        aggClassEst: 类别估计累计值

    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化权重
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 构建单层决策树
        # print("D:",D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha  # 存储弱学习算法权重
        weakClassArr.append(bestStump)  # 存储单层决策树
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst  # 计算类别估计累计值
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))  # 计算误差
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0: break  # 误差为0，退出循环
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数
    Args:
        datToClass:待分类样例
        classifierArr:训练好的分类器

    Returns:
        分类结果

    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):  # 遍历所有分类器，进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    """
    绘制ROC
    Args:
        predStrengths: 分类器的预测强度
        classLabels: 类别

    Returns:

    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    cur = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0  # 用于计算AUC
    numPosClas = np.sum(np.array(classLabels) == 1.0)  # 统计正类的数量
    yStep = 1 / float(numPosClas)  # y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)  # x轴步长

    sortedIndicies = predStrengths.argsort()  # 预测强度排序,从低到高
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]  # 高度累加
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')  # 绘制ROC
        cur = (cur[0] - delX, cur[1] - delY)  # 更新绘制光标的位置
    ax.plot([0, 1], [0, 1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties=font)
    plt.xlabel('假阳率', FontProperties=font)
    plt.ylabel('真阳率', FontProperties=font)
    ax.axis([0, 1, 0, 1])
    print('AUC面积为:', ySum * xStep)  # 计算AUC
    plt.show()


if __name__ == '__main__':
    # AdaBoost
    # dataArr, classLabels = loadSimpData()
    # weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    # print(adaClassify([[0, 0], [5, 5]], weakClassArr))

    # 马疝病的预测
    dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr, 50)
    plotROC(aggClassEst.T, LabelArr)

"""
集成方法: ensemble method（元算法: meta algorithm） 概述:

    ·概念：是对其他算法进行组合的一种形式。
    ·通俗来说： 当做重要决定时，大家可能都会考虑吸取多个专家而不只是一个人的意见。 机器学习处理问题时又何尝不是如此？ 这就是集成方法背后的思想。
    
    ·集成方法：
    
        a. 投票选举(bagging: 自举汇聚法 bootstrap aggregating): 是基于数据随机重抽样分类器构造的方法
        b. 再学习(boosting): 是基于所有分类器的加权求和的方法

"""

"""
集成方法 场景:

    目前 bagging 方法最流行的版本是: 随机森林(random forest)
    选男友：美女选择择偶对象的时候，会问几个闺蜜的建议，最后选择一个综合得分最高的一个作为男朋友
    
    目前 boosting 方法最流行的版本是: AdaBoost
    追女友：3个帅哥追同一个美女，第1个帅哥失败->(传授经验：姓名、家庭情况) 第2个帅哥失败->(传授经验：兴趣爱好、性格特点) 第3个帅哥成功
"""

"""
bagging 和 boosting 区别:

    1.bagging 是一种与 boosting 很类似的技术, 所使用的多个分类器的类型（数据量和特征量）都是一致的。
    2.bagging 是由不同的分类器（1.数据随机化 2.特征随机化）经过训练，综合得出的出现最多分类结果；
      boosting 是通过调整已有分类器错分的那些数据来获得新的分类器，得出目前最优的结果。
    3.bagging 中的分类器权重是相等的；而 boosting 中的分类器加权求和，所以权重并不相等，每个权重代表的是其对应
      分类器在上一轮迭代中的成功度。
"""


"""
AdaBoost 原理

    AdaBoost 工作原理
        ·见图adaboost_illustration.png
        
    AdaBoost 开发流程
        
        ·收集数据：可以使用任意方法
        ·准备数据：依赖于所使用的弱分类器类型，本章使用的是单层决策树，这种分类器可以处理任何数据类型。
                   当然也可以使用任意分类器作为弱分类器，第2章到第6章中的任一分类器都可以充当弱分类器。
                   作为弱分类器，简单分类器的效果更好。
        ·分析数据：可以使用任意方法。
        ·训练算法：AdaBoost 的大部分时间都用在训练上，分类器将多次在同一数据集上训练弱分类器。
        ·测试算法：计算分类的错误率。
        ·使用算法：通SVM一样，AdaBoost 预测两个类别中的一个。如果想把它应用到多个类别的场景，那么就要像多类 SVM 中的做法一样对 AdaBoost 进行修改。
        
    AdaBoost 算法特点
        
        * 优点：泛化（由具体的、个别的扩大为一般的）错误率低，易编码，可以应用在大部分分类器上，无参数调节。
        * 缺点：对离群点敏感。
        * 适用数据类型：数值型和标称型数据。
"""