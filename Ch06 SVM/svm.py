#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 17:31
# @Author  : zhcf1ess
# @Site    : 
# @File    : svm.py
# @Software: PyCharm
import numpy as np
from time import sleep


def loadDataSet(fileName):
    '''
    对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
    :param fileName: 文件名
    :return: dataMat  特征矩阵
             labelMat 类标签
    '''
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    '''
    调整aj的值，使aj处于 L<=aj<=H
    :param aj: 目标值
    :param H: 最大值
    :param L: 最小值
    :return: 目标值
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    SMO算法的简单实现
    :param dataMatIn: 特征集合
    :param classLabels: 类别标签
    :param C: 松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
              控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
              可以通过调节该参数达到不同的结果。
    :param toler: 容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
    :param maxIter: 退出前最大的循环次数
    :return: b       模型的常量值
             alphas  拉格朗日乘子
    '''
    dataMatrix = np.mat(dataMatIn)
    # 矩阵转置 和 .T 一样的功能
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)

    # 初始化 b和alphas(alpha有点类似权重值。)
    b = 0
    alphas = np.mat(np.zeros((m, 1)))

    # 没有任何alpha改变的情况下遍历数据的次数
    iter = 0
    while (iter < maxIter):
        # w = calcWs(alphas, dataMatIn, classLabels)
        # print("w:", w)

        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):
            # print 'alphas=', alphas
            # print 'labelMat=', labelMat
            # print 'multiply(alphas, labelMat)=', multiply(alphas, labelMat)
            # 我们预测的类别 y[i] = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*lable[n]*x[n]
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 预测结果与真实结果比对，计算误差Ei
            Ei = fXi - float(labelMat[i])

            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            '''
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):

                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                j = selectJrand(i, m)
                # 预测j的结果
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果相同，就没发优化了
                if L == H:
                    print("L==H")
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                      - dataMatrix[i, :] * dataMatrix[i, :].T \
                      - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 计算出一个新的alphas[j]值
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 并使用辅助函数，以及L和H对其进行调整
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
                # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
        # 知道更新完毕后，iter次循环无变化，才推出循环。
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


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


def calcEk(oS, k):
    '''
    求 Ek误差：预测值-真实值的差
    该过程在完整版的SMO算法中陪出现次数较多，因此将其单独作为一个方法
    :param oS: optStruct对象
    :param k: 具体的某一行
    :return: 预测结果与真实结果比对，计算误差Ek
    '''
    fXk = np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def loadImages(dirName):
    '''
    加载图片文件
    :param dirName: 图片文件的储存路径
    :return:
    '''
    from os import listdir
    hwLabels = []
    print(dirName)
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def kernelTrans(X, A, kTup):
    '''
    计算核空间 将数据转化到高维空间
    :param X: dataMatIn数据集
    :param A: dataMatIn数据集的第i行的数据
    :param kTup: 核函数的信息
    :return:
    '''
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        # linear kernel:   m*n * n*1 = m*1
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        K = np.exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    '''
    建立的数据结构来保存所有的重要值
    '''

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        '''
        初始化类变量
        :param dataMatIn: 数据集
        :param classLabels: 类别标签
        :param C: 松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
                  控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
                  可以通过调节该参数达到不同的结果。
        :param toler: 容错率
        :param kTup: 包含核函数信息的元组
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler

        # 数据的行数
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0

        # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
        self.eCache = np.mat(np.zeros((self.m, 2)))

        # m行m列的矩阵
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    '''
    返回最优的j和Ej(内循环的启发式方法。)
    :param i: 具体的第i一行
    :param oS: optStruct对象
    :param Ei: 预测结果与真实结果比对，计算误差Ei
    :return: j  随机选出的第j一行
             Ej 预测结果与真实结果比对，计算误差Ej
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。
    oS.eCache[i] = [1, Ei]

    # print('oS.eCache[%s]=%s' % (i, oS.eCache[i]))
    # print('oS.eCache[:, 0].A=%s' % oS.eCache[:, 0].A.T)
    # """
    # # 返回非0的：行列值
    # nonzero(oS.eCache[:, 0].A)= (
    #     行： array([ 0,  2,  4,  5,  8, 10, 17, 18, 20, 21, 23, 25, 26, 29, 30, 39, 46,52, 54, 55, 62, 69, 70, 76, 79, 82, 94, 97]),
    #     列： array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0])
    # )
    # """
    # print('nonzero(oS.eCache[:, 0].A)=', nonzero(oS.eCache[:, 0].A))
    # # 取行的list
    # print('nonzero(oS.eCache[:, 0].A)[0]=', nonzero(oS.eCache[:, 0].A)[0])
    # 非零E值的行的list列表，所对应的alpha值
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # 在所有的值上进行循环，并选择其中使得改变最大的那个值
            if k == i:
                continue  # don't calc for i, waste of time

            # 求 Ek误差：预测值-真实值的差
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                # 选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # 如果是第一次循环，则随机选择一个alpha值
        j = selectJrand(i, oS.m)

        # 求 Ek误差：预测值-真实值的差
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    '''
    计算误差值并存入缓存中
    (在对alpha值进行优化之后会用到这个值)
    :param oS: optStruct对象
    :param k: 某一列的行号
    :return:
    '''
    # 求 误差：预测值-真实值的差
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    '''
    内循环代码
    :param i: 具体的某一行
    :param oS: optStruct对象
    :return:  0   找不到最优的值
              1   找到了最优的值，并且oS.Cache到缓存中
    '''
    # 求 Ek误差：预测值-真实值的差
    Ei = calcEk(oS, i)

    # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
    # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
    # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
    '''
    # 检验训练样本(xi, yi)是否满足KKT条件
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化。效果更明显
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算出一个新的alphas[j]值
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            # print("j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''
    完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
              控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
              可以通过调节该参数达到不同的结果。
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :param kTup: 包含核函数信息的元组
    :return: b       模型的常量值
             alphas  拉格朗日乘子
    '''
    # 创建一个 optStruct 对象
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


'''
SVM工作原理:
    
    1.寻找最大分类间距
    2.转而通过拉格朗日函数求优化的问题
    3.数据可以通过画一条直线就可以将它们完全分开，这组数据叫线性可分(linearly separable)数据，而这条分隔直线称为
      分隔超平面(separating hyperplane)。
    4.如果数据集上升到1024维呢？那么需要1023维来分隔数据集，也就说需要N-1维的对象来分隔，这个对象叫做超平面(hyperlane)，
      也就是分类的决策边界。

'''

'''
SMO高效优化算法:

    SMO用途：用于训练 SVM
    SMO目标：求出一系列 alpha 和 b,一旦求出 alpha，就很容易计算出权重向量 w 并得到分隔超平面。
    SMO思想：是将大优化问题分解为多个小优化问题来求解的。
    SMO原理：每次循环选择两个 alpha 进行优化处理，一旦找出一对合适的 alpha，那么就增大一个同时减少一个。
                ·这里指的合适必须要符合一定的条件
                    a.这两个 alpha 必须要在间隔边界之外
                    b.这两个 alpha 还没有进行过区间化处理或者不在边界上。
                ·之所以要同时改变2个 alpha；原因是我们有一个约束条件： ∑mi=1ai⋅labeli=0；
                  如果只是修改一个 alpha，很可能导致约束条件失效。
                  
SMO 伪代码：

        创建一个 alpha 向量并将其初始化为0向量
        当迭代次数小于最大迭代次数时(外循环)
            对数据集中的每个数据向量(内循环)：
                如果该数据向量可以被优化
                    随机选择另外一个数据向量
                        同时优化这两个向量
                        如果两个向量都不能被优化，退出内循环
                如果所有向量都没被优化，增加迭代数目，继续下一次循环
'''

'''
svm开发流程:

    ·收集数据：可以使用任意方法。
    ·准备数据：需要数值型数据。
    ·分析数据：有助于可视化分隔超平面。
    ·训练算法：SVM的大部分时间都源自训练，该过程主要实现两个参数的调优。
    ·测试算法：十分简单的计算过程就可以实现。
    ·使用算法：几乎所有分类问题都可以使用SVM，值得一提的是，SVM本身是一个二类分类器，对多类问题应用SVM需要对代码做一些修改。
'''

'''
svm算法特点:
     
    ·优点：泛化错误率低，计算开销不大，结果易解释。
    ·缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二类问题。
    ·适用数据类型：数值型和标称型数据。

'''
