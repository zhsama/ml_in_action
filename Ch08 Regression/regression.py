#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/16 14:56
# @Author  : zhcf1ess
# @Site    : 
# @File    : regression.py
# @Software: PyCharm
import numpy as np
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
from time import sleep
import bs4
from bs4 import BeautifulSoup
import json
from urllib import request


def loadDataSet(fileName):
    """ 加载数据
        解析以tab键分隔的文件中的浮点数
    Returns：
        dataMat ：  feature 对应的数据集
        labelMat ： feature 对应的分类标签，即类别标签
    """
    # 获取样本特征的总数，不算最后的目标变量
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # 读取每一行
        lineArr = []
        # 删除一行中以tab分隔的数据前后的空白符号
        curLine = line.strip().split('\t')
        # i 从0到2，不包括2
        for i in range(numFeat):
            # 将数据添加到lineArr List中，每一行数据测试数据组成一个行向量
            lineArr.append(float(curLine[i]))
            # 将测试数据的输入数据部分存储到dataMat 的List中
        dataMat.append(lineArr)
        # 将每一行的最后一个数据，即类别，或者叫目标变量存储到labelMat List中
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    '''
    Description：
        线性回归
    Args:
        xArr ：输入的样本数据，包含每个样本数据的 feature
        yArr ：对应于输入数据的类别标签，也就是每个样本对应的目标变量
    Returns:
        ws：回归系数
    '''

    # mat()函数将xArr，yArr转换为矩阵 mat().T 代表的是对矩阵进行转置操作
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    xTx = xMat.T * xMat
    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法进行接下来的运算
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 最小二乘法
    # http://www.apache.wiki/pages/viewpage.action?pageId=5505133
    # 书中的公式，求得w的最优解
    ws = xTx.I * (xMat.T * yMat)
    return ws


def regression1(filename):
    '''
    测试线性回归
    Args:
        filename: 输入文件

    Returns:

    '''
    xArr, yArr = loadDataSet(filename)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    ws = standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)  # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
    ax.scatter([xMat[:, 1].flatten()], [yMat.T[:, 0].flatten().A[0]])  # scatter 的x是xMat中的第二列，y是yMat的第一列
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
    Args：
        testPoint：样本点
        xArr：样本的特征数据，即 feature
        yArr：每个样本对应的类别标签，即目标变量
        k: 关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关

    Returns:
        testPoint * ws：数据点与具有权重的系数相乘得到的预测点

    Notes:
        这其中会用到计算权重的公式，w = e^((x^((i))-x) / -2k^2)
        理解：x为某个预测点，x^((i))为样本点，样本点距离预测点越近，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）。
              关于预测点的选取，在我的代码中取的是样本点。其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差。
        算法思路：假设预测点取样本点中的第i个样本点（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
                  也就可以计算出每个样本贡献误差的权值，可以看出w是一个有m个元素的向量（写成对角阵形式）。
    '''
    # mat() 函数是将array转换为矩阵的函数， mat().T 是转换为矩阵之后，再进行转置操作
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 获得xMat矩阵的行数
    m = np.shape(xMat)[0]
    # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重
    weights = np.mat(np.eye((m)))
    for j in range(m):
        # testPoint 的形式是 一个行向量的形式
        # 计算 testPoint 与输入样本点之间的距离，然后下面计算出每个样本贡献误差的权值
        diffMat = testPoint - xMat[j, :]
        # k控制衰减的速度
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    # 根据矩阵乘法计算 xTx ，其中的 weights 矩阵是样本点对应的权重矩阵
    xTx = xMat.T * (weights * xMat)
    # 判断矩阵可否取逆
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 计算出回归系数的一个估计
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
    测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
    Args：
        testArr：测试所用的所有样本点
        xArr：样本的特征数据，即 feature
        yArr：每个样本对应的类别标签，即目标变量
        k：控制核函数的衰减速率

    Returns：
        yHat：预测点的估计值
    '''
    # 得到样本点的总数
    m = np.shape(testArr)[0]
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = np.zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    # 返回估计值
    return yHat


def lwlrTestPlot(xArr, yArr, k=1.0):
    '''
    首先将 X 排序，其余的都与lwlrTest相同，这样更容易绘图
    Args：
        xArr：样本的特征数据，即 feature
        yArr：每个样本对应的类别标签，即目标变量，实际值
        k：控制核函数的衰减速率的有关参数，这里设定的是常量值 1
    Return：
        yHat：样本点的估计值
        xCopy：xArr的复制
    '''
    # 生成一个与目标变量数目相同的 0 向量
    yHat = np.zeros(np.shape(yArr))
    # 将 xArr 转换为 矩阵形式
    xCopy = np.mat(xArr)
    # 排序
    xCopy.sort(0)
    # 开始循环，为每个样本点进行局部加权线性回归，得到最终的目标变量估计值
    for i in range(np.shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


def regression2(filename, k):
    '''
    局部加权线性回归测试
    Args:
        filename: 输入文件
        k: 权重

    Returns:

    '''
    xArr, yArr = loadDataSet(filename)
    yHat = lwlrTest(xArr, xArr, yArr, k)
    xMat = np.mat(xArr)
    srtInd = xMat[:, 1].argsort(0)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter([xMat[:, 1].flatten().A[0]], [np.mat(yArr).T.flatten().A[0]], s=2, c='red')
    plt.show()


def rssError(yArr, yHatArr):
    '''
    计算分析预测误差的大小
    Args:
        yArr：真实的目标变量
        yHatArr：预测得到的估计值

    Returns:
        计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    这个函数实现了给定 lambda 下的岭回归求解。
    如果数据的特征比样本点还多，就不能再使用上面介绍的的线性回归和局部现行回归了，因为计算 (xTx)^(-1)会出现错误。
    如果特征比样本点还多（n > m），也就是说，输入数据的矩阵x不是满秩矩阵。非满秩矩阵在求逆时会出现问题。
    为了解决这个问题，我们下边讲一下：岭回归，这是我们要讲的第一种缩减方法。
    Args：
        xMat：样本的特征数据，即 feature
        yMat：每个样本对应的类别标签，即目标变量，实际值
        lam：引入的一个λ值，使得矩阵非奇异
    Returns：
        经过岭回归公式计算得到的回归系数
    '''

    xTx = xMat.T * xMat
    # 岭回归就是在矩阵 xTx 上加一个 λI 从而使得矩阵非奇异，进而能对 xTx + λI 求逆
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    # 检查行列式是否为零，即矩阵是否可逆，行列式为0的话就不可逆，不为0的话就是可逆。
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    '''
    函数 ridgeTest() 用于在一组 λ 上测试结果
    Args：
        xArr：样本数据的特征，即 feature
        yArr：样本数据的类别标签，即真实数据
    Returns：
        wMat：将所有的回归系数输出到一个矩阵并返回
    '''

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 计算Y的均值
    yMean = np.mean(yMat, 0)
    # Y的所有的特征减去均值
    yMat = yMat - yMean
    # 标准化 x，计算 xMat 平均值
    xMeans = np.mean(xMat, 0)
    # 然后计算 X的方差
    xVar = np.var(xMat, 0)
    # 所有特征都减去各自的均值并除以方差
    xMat = (xMat - xMeans) / xVar
    # 可以在 30 个不同的 lambda 下调用 ridgeRegres() 函数。
    numTestPts = 30
    # 创建30 * m 的全部数据为0 的矩阵
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        # exp() 返回 e^x
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):
    '''
    按列进行规范化
    Args:
        xMat: 待规范矩阵

    Returns:
        inMat: 规范化后矩阵

    '''
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)  # 计算平均值然后减去它
    inVar = np.var(inMat, 0)  # 计算除以Xi的方差
    inMat = (inMat - inMeans) / inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    函数说明:前向逐步线性回归
    Args:
        xArr: x输入数据
        yArr: y预测数据
        eps: 每次迭代需要调整的步长
        numIt: 迭代次数

    Returns:
		returnMat: numIt次迭代的回归系数矩阵

    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean  # 也可以规则化ys但会得到更小的coef
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))  # 测试代码删除
    ws = np.zeros((n, 1))
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


def plotstageWiseMat():
    '''
    函数说明:绘制岭回归系数矩阵

    Returns:

    '''
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    xArr, yArr = loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


def abaloneTest():
    '''
    Desc:
        预测鲍鱼的年龄
    Args:
        None
    Returns:
        None
    '''
    # 加载数据
    abX, abY = loadDataSet("abalone.txt")
    # 使用不同的核进行预测
    oldyHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    oldyHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    oldyHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    print("old yHat01 error Size is :", rssError(abY[0:99], oldyHat01.T))
    print("old yHat1 error Size is :", rssError(abY[0:99], oldyHat1.T))
    print("old yHat10 error Size is :", rssError(abY[0:99], oldyHat10.T))

    # 打印出 不同的核预测值 与 新数据集（测试数据集）上的真实值之间的误差大小
    newyHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print("new yHat01 error Size is :", rssError(abY[0:99], newyHat01.T))
    newyHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print("new yHat1 error Size is :", rssError(abY[0:99], newyHat1.T))
    newyHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print("new yHat10 error Size is :", rssError(abY[0:99], newyHat10.T))

    # 使用简单的 线性回归 进行预测，与上面的计算进行比较
    standWs = standRegres(abX[0:99], abY[0:99])
    standyHat = np.mat(abX[100:199]) * standWs
    print("standRegress error Size is:", rssError(abY[100:199], standyHat.T.A))


if __name__ == '__main__':
    # regression1('ex0.txt')
    # regression1('ex1.txt')
    # regression2('ex0.txt', k=1)
    # regression2('ex1.txt', k=0.01)
    # regression2('ex1.txt', k=0.003)
    plotstageWiseMat()
'''
线性回归 工作原理
    
    1. 读入数据，将数据特征x、特征标签y存储在矩阵x、y中
    2. 验证 x^Tx 矩阵是否可逆
    3. 使用最小二乘法求得 回归系数 w 的最佳估计

线性回归 开发流程

    ·收集数据: 采用任意方法收集数据
    ·准备数据: 回归需要数值型数据，标称型数据将被转换成二值型数据
    ·分析数据: 绘出数据的可视化二维图将有助于对数据做出理解和分析，在采用缩减法求得新回归系数之后，可以将新拟合线绘在图上作为对比
    ·训练算法: 找到回归系数
    ·测试算法: 使用 R^2 或者预测值和数据的拟合度，来分析模型的效果
    ·使用算法: 使用回归，可以在给定输入的时候预测出一个数值，这是对分类方法的提升，因为这样可以预测连续型数据而不仅仅是离散的类别标签


线性回归 算法特点
    
    优点：结果易于理解，计算上不复杂。
    缺点：对非线性的数据拟合不好。
    适用于数据类型：数值型和标称型数据。

'''
