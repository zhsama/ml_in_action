#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 16:10
# @Author  : zhcf1ess
# @Site    : 
# @File    : plotIteration.py
# @Software: PyCharm
from numpy import *
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.pardir)
import logistic

"""
运行随机梯度上升算法，在数据集的一次遍历中回归系数与迭代次数的关系图。
回归系数经过大量迭代才能达到稳定值，并且仍然有局部的波动现象。
"""


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.5
    weights = ones(n)  # initialize to all ones
    weightsHistory = zeros((500 * m, n))
    for j in range(500):
        for i in range(m):
            h = logistic.sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            weightsHistory[j * m + i, :] = weights
    return weightsHistory


def stocGradAscent1(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    weightsHistory = zeros((40 * m, n))
    for j in range(40):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = logistic.sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            # print error
            weights = weights + alpha * error * dataMatrix[randIndex]
            weightsHistory[j * m + i, :] = weights
            del (dataIndex[randIndex])
    print(weights)
    return weightsHistory


dataMat, labelMat = logistic.loadDataSet()
dataArr = array(dataMat)
myHist = stocGradAscent1(dataArr, labelMat)

n = shape(dataArr)[0]  # number of points to create
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []

markers = []
colors = []

fig = plt.figure()
ax = fig.add_subplot(311)
type1 = ax.plot(myHist[:, 0])
plt.ylabel('X0')
ax = fig.add_subplot(312)
type1 = ax.plot(myHist[:, 1])
plt.ylabel('X1')
ax = fig.add_subplot(313)
type1 = ax.plot(myHist[:, 2])
plt.xlabel('iteration')
plt.ylabel('X2')
plt.show()
