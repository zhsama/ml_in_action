#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 9:24
# @Author  : zhcf1ess
# @Site    : 
# @File    : knn.py
# @Software: PyCharm


import numpy as np
import operator
import os

'''
优化约会网站的配对效果开发流程

    收集数据：提供文本文件
    准备数据：使用 Python 解析文本文件
    分析数据：使用 Matplotlib 画二维散点图
    训练算法：此步骤不适用于 k-近邻算法
    测试算法：使用海伦提供的部分数据作为测试样本。
            测试样本和非测试样本的区别在于：
                测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
    使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。
'''

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    KNN实现
    :param inX:     用于分类的输入向量是inX
    :param dataSet: 输入的训练样本集为dataSet
    :param labels:  标签向量为labels
    :param k:       k表示用于选择最近邻居的数目
    :return:        排序后的KNN输出
    '''
    # 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    # 选取距离最近的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对结果排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    '''
    文本记录转换为 NumPy
    :param filename: 待解析的文本路径
    :return: numpy对象=>数据矩阵 returnMat 和对应的类别 classLabelVector
    '''
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 获得文件中的数据行的行数
    # 生成对应的空矩阵
    # 创建一个2维矩阵用于存放训练样本数据，一共有n行，每一行存放3个数据
    # 例如：zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
    returnMat = np.zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # 创建一个1维数组用于存放训练样本标签
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        listFromLine = line.split('\t')  # 以 '\t' 切割字符串
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]  # 把分割好的数据放至数据集 其中index是该样本数据的下标 就是放到第几行
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector


# 使用 Matplotlib 画二维散点图
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
# plt.show()


def autoNorm(dataSet):
    '''
    归一化特征值
    归一化是一个统一权重的过程
    归一化的目的就是使得预处理的数据被限定在一定的范围内（比如[0,1]或者[-1,1]），从而消除奇异样本数据导致的不良影响
    :param dataSet: 输入数组
    :return: 归一化后的新矩阵
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    '''
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)  # axis=0 从列中选取最小值
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    # normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    '''
    训练数据 输出训练结果
    :return: 正确率 错误率等...
    '''
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('input/2.KNN/datingTestSet2.txt')  # load data setfrom file
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


'''
手写数字识别系统开发流程

    收集数据：提供文本文件。
    准备数据：编写函数 img2vector(), 将图像格式转换为分类器使用的向量格式
    分析数据：在 Python 命令提示符中检查数据，确保它符合要求
    训练算法：此步骤不适用于 KNN
    测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的
             区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，
             则标记为一个错误
    使用算法：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取
             数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统
'''


def img2vector(filename):
    '''
    把输入图片转换为矩阵
    :param filename: 输入图片路径
    :return: 图片转换矩阵
    '''
    # 创建1×1024的NumPy数组，然后打开给定的文件
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    # 循环读出文件的前32行
    for i in range(32):
        lineStr = fr.readline()
        # 将每行的头32个字符值存储在NumPy数组中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    '''
    手写识别
    :return:
    '''
    # 1.导入训练数据
    hwLabels = []
    trainingFileList = os.listdir('input/2.KNN/trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('input/2.KNN/trainingDigits/%s' % fileNameStr)

    # 2.导入测试数据
    testFileList = os.listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('input/2.KNN/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


'''
KNN 概述:

    k 近邻算法的输入为实例的特征向量，对应于特征空间的点；输出为实例的类别，可以取多类。
    k 近邻算法假设给定一个训练数据集，其中的实例类别已定。
    分类时，对新的实例，根据其 k 个最近邻的训练实例的类别，通过多数表决等方式进行预测。
    因此，k近邻算法不具有显式的学习过程。

    k 近邻算法实际上利用训练数据集对特征向量空间进行划分，并作为其分类的“模型”。 
    k值的选择、距离度量以及分类决策规则是k近邻算法的三个基本要素。
'''

'''
KNN 工作原理:

    1.假设有一个带有标签的样本数据集（训练样本集），其中包含每条数据与所属分类的对应关系。
    2.输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较。
        a.计算新数据与样本数据集中每条数据的距离。
        b.对求得的所有距离进行排序（从小到大，越小表示越相似）。
        c.取前 k （k 一般小于等于 20 ）个样本数据对应的分类标签。
    3.求 k 个数据中出现次数最多的分类标签作为新数据的分类。
'''

'''
KNN 开发流程:

    收集数据：任何方法
    准备数据：距离计算所需要的数值，最好是结构化的数据格式
    分析数据：任何方法
    训练算法：此步骤不适用于 k-近邻算法
    测试算法：计算错误率
    使用算法：输入样本数据和结构化的输出结果，然后运行 k-近邻算法判断输入数据分类属于哪个分类，最后对计算出的分类执行后续处理
'''

'''
KNN 算法特点:

    优点：精度高、对异常值不敏感、无数据输入假定
    缺点：计算复杂度高、空间复杂度高
    适用数据范围：数值型和标称型
'''

'''
要素发现:
    ·k 值的选择:

        ·k 值的选择会对 k 近邻算法的结果产生重大的影响。

        ·如果选择较小的 k 值，就相当于用较小的邻域中的训练实例进行预测，“学习”的近似误差（approximation error）会减小，
          只有与输入实例较近的（相似的）训练实例才会对预测结果起作用。但缺点是“学习”的估计误差（estimation error）会增大，
          预测结果会对近邻的实例点非常敏感。如果邻近的实例点恰巧是噪声，预测就会出错。换句话说，k 值的减小就意味着整体模型变得
          复杂，容易发生过拟合。

        ·如果选择较大的 k 值，就相当于用较大的邻域中的训练实例进行预测。其优点是可以减少学习的估计误差。但缺点是学习的近似
          误差会增大。这时与输入实例较远的（不相似的）训练实例也会对预测起作用，使预测发生错误。 k 值的增大就意味着整体的模型
          变得简单。

        ·近似误差和估计误差，请看这里：https://www.zhihu.com/question/60793482

    ·距离度量

        ·特征空间中两个实例点的距离是两个实例点相似程度的反映。

        ·k 近邻模型的特征空间一般是 n 维实数向量空间Rn向量空间 。使用的距离是欧氏距离，但也可以是其他距离，
        如更一般的 Lp距离 距离，或者 Minkowski 距离。

    ·分类决策规则
        ·k 近邻算法中的分类决策规则往往是多数表决，即由输入实例的 k 个邻近的训练实例中的多数类决定输入实例的类。
'''
