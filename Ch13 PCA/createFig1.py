'''
Created on Jun 1, 2011

@author: Peter
'''
import os,sys
import numpy as np
sys.path.append(os.pardir)
import matplotlib.pyplot as plt

n = 1000 #number of points to create
xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers =[]
colors =[]
fw = open('../data/ch13/testSet.txt','w')
for i in range(n):
    [r0,r1] = np.random.standard_normal(2)
    fFlyer = r0 + 9.0
    tats = 1.0*r1 + fFlyer + 0
    xcord0.append(fFlyer)
    ycord0.append(tats)
    fw.write("%f\t%f\n" % (fFlyer, tats))

fw.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0,ycord0, marker='^', s=90)
plt.xlabel('hours of direct sunlight')
plt.ylabel('liters of water')
plt.show()