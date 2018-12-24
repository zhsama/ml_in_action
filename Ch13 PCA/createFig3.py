'''
Created on Jun 1, 2011

@author: Peter
'''
import matplotlib.pyplot as plt
import os, sys
import numpy as np

sys.path.append(os.pardir)
import pca

n = 1000  # number of points to create
xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []
markers = []
colors = []
fw = open('../data/ch13/testSet3.txt', 'w')
for i in range(n):
    groupNum = int(3 * np.random.uniform())
    [r0, r1] = np.random.standard_normal(2)
    if groupNum == 0:
        x = r0 + 16.0
        y = 1.0 * r1 + x
        xcord0.append(x)
        ycord0.append(y)
    elif groupNum == 1:
        x = r0 + 8.0
        y = 1.0 * r1 + x
        xcord1.append(x)
        ycord1.append(y)
    elif groupNum == 2:
        x = r0 + 0.0
        y = 1.0 * r1 + x
        xcord2.append(x)
        ycord2.append(y)
    fw.write("%f\t%f\t%d\n" % (x, y, groupNum))

fw.close()
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(xcord0, ycord0, marker='^', s=90)
ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
ax.scatter(xcord2, ycord2, marker='v', s=50, c='yellow')
ax = fig.add_subplot(212)
myDat = pca.loadDataSet('../data/ch13/testSet3.txt')
lowDDat, reconDat = pca.pca(myDat[:, 0:2], 1)
label0Mat = lowDDat[np.nonzero(myDat[:, 2] == 0)[0], :2][0]  # get the items with label 0
label1Mat = lowDDat[np.nonzero(myDat[:, 2] == 1)[0], :2][0]  # get the items with label 1
label2Mat = lowDDat[np.nonzero(myDat[:, 2] == 2)[0], :2][0]  # get the items with label 2
ax.scatter(label0Mat[:, 0].tolist(), np.zeros(np.shape(label0Mat)[0]), marker='^', s=90)
ax.scatter(label1Mat[:, 0].tolist(), np.zeros(np.shape(label1Mat)[0]), marker='o', s=50, c='red')
ax.scatter(label2Mat[:, 0].tolist(), np.zeros(np.shape(label2Mat)[0]), marker='v', s=50, c='yellow')
plt.show()
