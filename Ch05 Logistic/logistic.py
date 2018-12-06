#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 8:55
# @Author  : zhcf1ess
# @Site    : 
# @File    : logistic.py
# @Software: PyCharm
import matplotlib.pyplot as plt

import numpy as np


###### 使用 Logistic 回归在简单数据集上的分类 ######

# 从文件读取数据
def loadDataSet():
    '''
    从文件读取数据
    :return:
    '''
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid函数
def sigmoid(inX):
    r = 1.0 / (np.exp(-inX) + 1)
    return r


# Logistic 回归梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    '''
    Logistic 回归梯度上升优化算法
    :param dataMatIn: 一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
    :param classLabels:类别标签 它是一个 1*100 的行向量
                       为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给labelMat
    :return: 回归系数
    '''
    # 转化为矩阵[[1,1,2],[1,1,2]....]
    dataMatrix = np.mat(dataMatIn)  # 转换为 NumPy 矩阵
    # 转化为矩阵[[0,1,0,1,0,1.....]]，并转制[[0],[1],[0].....]
    # transpose() 行列转置函数
    # 将行向量转化为列向量   =>  矩阵的转置
    labelMat = np.mat(classLabels).transpose()  # 首先将数组转换为 NumPy 矩阵，然后再将行向量转置为列向量
    # m->数据量，样本数 n->特征数
    m, n = np.shape(dataMatrix)
    # print m, n, '__'*10, shape(dataMatrix.transpose()), '__'*100
    # alpha代表向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 生成一个长度和特征数相同的矩阵，此处n为3 -> [[1],[1],[1]]
    # weights 代表回归系数， 此处的 ones((n,1)) 创建一个长度和特征数相同的矩阵，其中的数全部都是 1
    weights = np.ones((n, 1))
    for k in range(maxCycles):  # heavy on matrix operations
        # m*3 的矩阵 * 3*1 的单位矩阵 ＝ m*1的矩阵
        # 那么乘上单位矩阵的意义，就代表：通过公式得到的理论值
        # 参考地址： 矩阵乘法的本质是什么？ https://www.zhihu.com/question/21351965/answer/31050145
        # print 'dataMatrix====', dataMatrix
        # print 'weights====', weights
        # n*3   *  3*1  = n*1
        h = sigmoid(dataMatrix * weights)  # 矩阵乘法
        labelMat.dtype = 'float32'
        # print 'hhhhhhh====', h
        # labelMat是实际值
        error = (labelMat - h)  # 向量相减
        # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，最后得出 x1,x2,xn的系数的偏移量
        # alpha * dataMatrix.transpose() * error 这玩意就是梯度
        weights = weights + alpha * dataMatrix.transpose() * error  # 矩阵乘法，最后得到回归系数
    return np.array(weights)


# 使用 Logistic 回归进行分类
def testLR():
    '''
    使用 Logistic 回归进行分类
    :return: 分类结果
    '''
    # 1.收集并准备数据
    dataMat, labelMat = loadDataSet('testSet.txt')
    # print(dataMat, '---\n', labelMat)

    # 2.训练模型，  f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    # 因为数组没有是复制n份， array的乘法就是乘法
    dataArr = np.array(dataMat)
    # print(dataArr)
    weights = gradAscent(dataArr, labelMat)
    # weights = stocGradAscent0(dataArr, labelMat)
    # weights = stocGradAscent1(dataArr, labelMat)
    # print('*'*30, weights)

    # 数据可视化
    plotBestFit(dataArr, labelMat, weights)


# 随机梯度上升
def stocGradAscent0(dataMatrix, classLabels):
    '''
    随机梯度上升
    梯度上升优化算法在每次更新数据集时都需要遍历整个数据集，计算复杂都较高
    随机梯度上升一次只用一个样本点来更新回归系数
    :param dataMatIn: 一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
    :param classLabels:类别标签 它是一个 1*100 的行向量
                       为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给labelMat
    :return: 回归系数
    '''
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    # n*1的矩阵
    # 函数ones创建一个全1的数组
    weights = np.ones(n)  # 初始化长度为n的数组，元素全部为 1
    # weights.dtype = 'float32'
    for i in range(m):
        # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataMatrix[i] * weights))
        # print 'dataMatrix[i]===', dataMatrix[i]
        # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = classLabels[i] - h
        # 0.01*(1*1)*(1*n)
        # print(weights, "*" * 10, dataMatrix[i], "*" * 10, error)
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 数据可视化
def plotBestFit(dataArr, labelMat, weights):
    '''
    数据可视化展示
    :param dataArr: 样本数据的特征
    :param labelMat: 样本数据的类别标签，即目标变量
    :param weights: 回归系数
    :return:
    '''
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    """
    y的由来:

        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])w0*x0+w1*x1+w2*x2=f(x)
        x0最开始就设置为1，x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
        所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# 随机梯度上升算法（随机化）
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    随机梯度上升算法（随机化）
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)  # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
    # 随机梯度, 循环150,观察是否收敛
    for j in range(numIter):
        # [0, 1, 2 .. m-1]
        dataIndex = list(range(m))
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            # print weights, '__h=%s' % h, '__'*20, alpha, '__'*20, error, '__'*20, dataMatrix[randIndex]
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


###### 从疝气病症预测病马的死亡率 ######

# 分类函数，根据回归系数和特征向量来计算 Sigmoid的值
def classifyVector(inX, weights):
    '''
    分类函数，根据回归系数和特征向量来计算 Sigmoid的值
    大于0.5函数返回1，否则返回0
    :param inX: 特征向量，features
    :param weights: 根据梯度下降/随机梯度下降 计算得到的回归系数
    :return: 如果 prob 计算大于 0.5 函数返回 1 否则返回 0
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 打开测试集和训练集,并对数据进行格式化处理
def colicTest():
    '''
    打开测试集和训练集,并对数据进行格式化处理
    :return: 分类错误率
    '''
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 解析训练数据集中的数据特征和Labels
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用 改进后的 随机梯度下降算法 求得在此数据集上的最佳回归系数 trainWeights
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


# 调用 colicTest() 10次并求结果的平均值
def multiTest():
    '''
    调用 colicTest() 10次并求结果的平均值
    :return:
    '''
    numTests = 100
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


'''
Logistic 回归 工作原理

    每个回归系数初始化为 1
    重复 R 次:
        计算整个数据集的梯度
        使用 步长 x 梯度 更新回归系数的向量
    返回回归系数
'''

'''
Logistic 回归 开发流程:

    ·收集数据: 采用任意方法收集数据
    ·准备数据: 由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
    ·分析数据: 采用任意方法对数据进行分析。
    ·训练算法: 大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
    ·测试算法: 一旦训练步骤完成，分类将会很快。
    ·使用算法: 首先，我们需要输入一些数据，并将其转换成对应的结构化数值；接着，基于训练好的回归系数就可以
      对这些数值进行简单的回归计算，判定它们属于哪个类别；在这之后，我们就可以在输出的类别上做一些其他分析工作。
'''

'''
Logistic 回归 算法特点:
    
    ·优点: 计算代价不高，易于理解和实现。
    ·缺点: 容易欠拟合，分类精度可能不高。
    ·适用数据类型: 数值型和标称型数据。
'''
