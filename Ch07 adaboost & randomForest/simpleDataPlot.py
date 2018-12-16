#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/16 9:38
# @Author  : zhcf1ess
# @Site    :
# @File    : simpleDataPlot.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
datMat = np.matrix([[ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]


xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers =[]
colors =[]

for i in range(len(classLabels)):
    if classLabels[i]==1.0:
        xcord1.append(datMat[i,0]), ycord1.append(datMat[i,1])
    else:
        xcord0.append(datMat[i,0]), ycord0.append(datMat[i,1])
fig = plt.figure()
ax = fig.add_subplot(111)       
ax.scatter(xcord0,ycord0, marker='s', s=90)
ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
plt.title('decision stump test data')
plt.show()
