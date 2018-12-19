#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/30 17:01
# @Author  : zhcf1ess
# @Site    : 
# @File    : trees.py
# @Software: PyCharm
from math import log
import operator
import decisionTreePlot as dtPlot


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 计算数据集中实例的总数
    labelCounts = {}  # 键值是最后一列的数值 记录当前类别出现的次数
    # 使用所有类标签的发生频率,计算类别出现的概率
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log2(n)=>ln(n)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 输入数据集
    :param axis: 用于划分数据集的特征值
    :param value: 需要返回的特征值
    :return: 划分后的数据集
    """
    retDataSet = []  # 创建新列表
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 切断用于分割的轴
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    将遍历整个数据集，循环计算香农熵和splitDataSet()函数，找到最好的特征划分方式
    :param dataSet: 输入数据集
    :return: 最佳切分方法
    输入数据集要求:
        ①数据由列表组成 且数据长度相同
        ②数据最后一列是数据的离散属性(标签)
    """
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)  # 计算整个数据集的香农熵
    bestInfoGain = 0.0
    bestFeature = -1

    # 遍历所有特征值
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 创建的这个特性的所有标签的列表
        uniqueVals = set(featList)  # 设置唯一标示值

        newEntropy = 0.0  # 初始化信息熵
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy  # 计算信息增益

        # 计算最好的划分方式
        if (infoGain > bestInfoGain):  # 与目前的最优信息熵进行比较
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最好信息熵的序列号


def majorityCnt(classList):
    """
    返回出现次数最多的分类名称
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 输入数据集
    :param labels: 数据标签
    :return: 创建的树结构
    """
    classList = [example[-1] for example in dataSet]  # 获取输入数据的标签集合

    if classList.count(classList[0]) == len(classList):  # 类别标签相同则停止继续划分
        return classList[0]

    if len(dataSet[0]) == 1:  # 当没有下一个分类标签时停止划分
        return majorityCnt(classList)  # 返回出现次数最多的特征值
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 当前数据集选取的最好特征存储 得到列表包含的所有属性值
    bestFeatLabel = labels[bestFeat]  # 最好的特征标签
    myTree = {bestFeatLabel: {}}  # 储存树的数据结构的字典
    del (labels[bestFeat])  # 删除标签bestFeat

    # 设置唯一标识值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    #
    for value in uniqueVals:
        subLabels = labels[:]  # 复制现有标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    获取列表中的特征标签
    :param inputTree: 输入决策树
    :param featLabels: 特征标签列表
    :param testVec: 测试矩阵
    :return: 匹配结果
    """
    firstStr = list(inputTree.keys())[0]  # 获取第一个特征标签 # dict_keys(['flippers']) # dict_keys(['no surfacing'])
    secondDict = inputTree[firstStr]  # 获取第一个特征标签对应的字典(即根节点的子节点)
    featIndex = featLabels.index(firstStr)  # 特征标签在特征列表中的位置
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]  #
    if isinstance(valueOfFeat, dict):  # 若获取到的节点为字典(即还有子节点)
        classLabel = classify(valueOfFeat, featLabels, testVec)  # 递归调用
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    """
    储存决策树
    :param inputTree: 需储存的决策树
    :param filename: 储存文件名
    :return:
    """
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    """
    读取已保存的决策树
    :param filename: 决策树文件名
    :return: 决策树
    """
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def fishTest():
    """
    对动物是否是鱼类分类的测试函数，并将结果使用 matplotlib 画出来
    Args:
        None
    Returns:
        None
    """
    # 1.创建数据和结果标签
    myDat, labels = createDataSet()
    # print(myDat, labels)

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print('1---', splitDataSet(myDat, 0, 1))
    # print('0---', splitDataSet(myDat, 0, 0))

    # # 计算最好的信息增益的列
    # print(chooseBestFeatureToSplit(myDat))

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # 画图可视化展现
    dtPlot.createPlot(myTree)


def ContactLensesTest():
    """
    预测隐形眼镜的测试代码，并将结果画出来
    Args:
        none
    Returns:
        none
    """

    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('../data/ch03/lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    dtPlot.createPlot(lensesTree)


if __name__ == "__main__":
    fishTest()
    # ContactLensesTest()
"""
决策树概述

    ·决策树（Decision Tree）算法主要用来处理分类问题，是最经常使用的数据挖掘算法之一。
"""

"""
决策树相关概念:
    ·信息熵 & 信息增益:
    
        ·熵： 熵（entropy）指的是体系的混乱的程度，在不同的学科中也有引申出的更为具体的定义，是各领域十分重要的参量。
        ·信息熵（香农熵）： 是一种信息的度量方式，表示信息的混乱程度，也就是说：信息越有序，信息熵越低。例如：火柴有序放在火柴盒
          里，熵值很低，相反，熵值很高。
        ·信息增益： 在划分数据集前后信息发生的变化称为信息增益。
"""

"""
决策树工作原理:
    ·伪代码createBranch():
    
    检测数据集中的所有数据的分类标签是否相同:
    If so return 类标签
    Else:
        寻找划分数据集的最好特征（划分之后信息熵最小，也就是信息增益最大的特征）
        划分数据集
        创建分支节点
            for 每个划分的子集
                调用函数 createBranch （创建分支的函数）并增加返回结果到分支节点中
        return 分支节点
"""

"""
决策树开发流程:
    
    ·收集数据：可以使用任何方法。
    ·准备数据：树构造算法只适用于标称型数据，因此数值型数据必须离散化。
    ·分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
    ·训练算法：构造树的数据结构。
    ·测试算法：使用经验树计算错误率。
    ·使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义。
"""

"""
决策树算法特点:

    ·优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。
    
    ·缺点：可能会产生过度匹配问题。
    
    ·适用数据类型：数值型和标称型。
"""
