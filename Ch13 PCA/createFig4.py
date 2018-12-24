'''
Created on Jun 14, 2011

@author: Peter
'''
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.pardir)
import pca

dataMat = pca.replaceNanWithMean()

#below is a quick hack copied from pca.pca()
meanVals = np.mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals #remove mean
covMat = np.cov(meanRemoved, rowvar=False)
eigVals,eigVects = np.linalg.eig(np.mat(covMat))
eigValInd = np.argsort(eigVals)            #sort, sort goes smallest to largest
eigValInd = eigValInd[::-1]#reverse
sortedEigVals = eigVals[eigValInd]
total = sum(sortedEigVals)
varPercentage = sortedEigVals/total*100

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^')
plt.xlabel('Principal Component Number')
plt.ylabel('Percentage of Variance')
plt.show()