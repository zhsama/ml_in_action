'''
Created on Jun 1, 2011

@author: Peter
'''
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.pardir)
import pca

dataMat = pca.loadDataSet('../data/ch13/testSet.txt')
lowDMat, reconMat = pca.pca(dataMat, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].tolist(), dataMat[:,1].tolist(), marker='^', s=90)
ax.scatter(reconMat[:,0].tolist(), reconMat[:,1].tolist(), marker='o', s=50, c='red')
plt.show()