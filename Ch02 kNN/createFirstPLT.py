#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/30 16:52
# @Author  : zhcf1ess
# @Site    : 
# @File    : createFirstPLT.py
# @Software: PyCharm
from numpy import *
import knn
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = knn.file2matrix('../data/ch02/datingTestSet2.txt')
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
ax.axis([-2, 25, -0.2, 2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()
