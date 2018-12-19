#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/18 19:45
# @Author  : zhcf1ess
# @Site    : 
# @File    : regTrees.py
# @Software: PyCharm
import numpy as np


def binSplitDataSet(dataSet, feature, value):
    '''
    将数据集按照feature列的value进行二元切分
    在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回。
    Args:
        dataMat: 数据集
        feature: 待切分的特征列
        value: 特征列要比较的值
    Returns:
        mat0: 小于等于 value 的数据集在左边
        mat1: 大于 value 的数据集在右边

    '''
    # dataSet[:, feature] 取去每一行中，第1列的值(从0开始算)
    # nonzero(dataSet[:, feature] > value) 返回结果为true行的index下标
    mat0 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1
