#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/3 17:18
# @Author  : zhcf1ess
# @Site    : 
# @File    : treesPlot.py
# @Software: PyCharm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def createPlot(inTree):
    # createPlot.ax1 = plt.subplot(111, frameon=False)  # ticks for demo puropses
    # plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plotNode('叶子节点', (0.8, 0.1), (0.3, 0.5), leftNode)
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):
    '''
    计算叶子节点的数量
    :param myTree: 输入决策树
    :return: 叶子节点的数量
    '''
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试节点是否为字典 若是 递归调用该函数继续计算叶子节点
            numLeafs += getNumLeafs(secondDict[key])
        else:  # 若不是 该节点为叶子节点
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    '''
    计算决策树的层数
    :param myTree: 输入决策树
    :return: 决策树层数
    '''
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 判断该节点是否为字典 若是 递归调用继续计算
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:  # 若不是 层数为1
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth  # 计算最深的层数
    return maxDepth


def retrieveTree(i):
    '''
    测试层数和叶子节点计算函数
    :param i:
    :return:
    '''
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    '''
    在节点间添加信息
    :param cntrPt: 子节点
    :param parentPt: 父节点
    :param txtString: 添加信息
    :return:
    '''
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    # 计算子节点个数和层数
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)

    firstStr = list(myTree.keys())[0]  # 添加节点的标签
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 标记叶子节点的属性
    plotMidText(cntrPt, parentPt, nodeTxt)  # 在父子节点之间添加文字
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘图
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 减少y偏移量(即将y坐标下移)
    # 对节点进行判断
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 判断该节点是否为字典
            plotTree(secondDict[key], cntrPt, str(key))  # 对字典(即父节点)递归调用该函数
        else:  # 该节点非字典 即是叶子节点
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
