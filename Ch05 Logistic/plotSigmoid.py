#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 16:18
# @Author  : zhcf1ess
# @Site    : 
# @File    : plotSigmoid.py
# @Software: PyCharm
from pylab import *

t = arange(-60.0, 60.3, 0.1)
s = 1 / (1 + exp(-t))
ax = subplot(211)
ax.plot(t, s)
ax.axis([-5, 5, 0, 1])
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
ax = subplot(212)
ax.plot(t, s)
ax.axis([-60, 60, 0, 1])
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
show()
