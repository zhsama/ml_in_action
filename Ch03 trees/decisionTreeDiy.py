# author: xiaolinhan_daisy
# date: 2017/12/25
# site: YueJiaZhuang
from numpy import *


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [1, 1, 'yes'],
               [1, 1, 'no']]
    labels = ['round', 'red']
    return dataSet, labels


# 计算数据集的entropy
def calcEntropy(dataSet):
    totalNum = len(dataSet)
    labelNum = {}
    entropy = 0
    for data in dataSet:
        label = data[-1]
        if label in labelNum:
            labelNum[label] += 1
        else:
            labelNum[label] = 1

    for key in labelNum:
        p = labelNum[key] / totalNum
        entropy -= p * log2(p)
    return entropy


# 计算特征值的熵
def calcEntropyForFeature(featureList):
    totalNum = len(featureList)
    dataNum = {}
    entropy = 0
    for data in featureList:
        if data in dataNum:
            dataNum[data] += 1
        else:
            dataNum[data] = 1

    for key in dataNum:
        p = dataNum[key] / totalNum
        entropy -= p * log2(p)
    return entropy


# 选择最优划分属性ID3
def chooseBestFeatureID3(dataSet, labels):
    bestFeature = 0
    initialEntropy = calcEntropy(dataSet)
    biggestEntropyG = 0
    for i in range(len(labels)):
        currentEntropy = 0
        feature = [data[i] for data in dataSet]
        subSet = splitDataSetByFeature(i, dataSet)
        totalN = len(feature)
        for key in subSet:
            prob = len(subSet[key]) / totalN
            currentEntropy += prob * calcEntropy(subSet[key])
        entropyGain = initialEntropy - currentEntropy
        if (biggestEntropyG < entropyGain):
            biggestEntropyG = entropyGain
            bestFeature = i
    return bestFeature


# 选择最优划分属性C4.5
def chooseBestFeatureC45(dataSet, labels):
    bestFeature = 0
    initialEntropy = calcEntropy(dataSet)
    biggestEntropyGR = 0
    for i in range(len(labels)):
        currentEntropy = 0
        feature = [data[i] for data in dataSet]
        entropyFeature = calcEntropyForFeature(feature)
        subSet = splitDataSetByFeature(i, dataSet)
        totalN = len(feature)
        for key in subSet:
            prob = len(subSet[key]) / totalN
            currentEntropy += prob * calcEntropy(subSet[key])
        entropyGain = initialEntropy - currentEntropy
        entropyGainRatio = entropyGain / entropyFeature

        if (biggestEntropyGR < entropyGainRatio):
            biggestEntropyGR = entropyGainRatio
            bestFeature = i
    return bestFeature


def splitDataSetByFeature(i, dataSet):
    subSet = {}
    feature = [data[i] for data in dataSet]
    for j in range(len(feature)):
        if feature[j] not in subSet:
            subSet[feature[j]] = []

        splittedDataSet = dataSet[j][:i]
        splittedDataSet.extend(dataSet[j][i + 1:])
        subSet[feature[j]].append(splittedDataSet)
    return subSet


def checkIsOneCateg(newDataSet):
    flag = False
    categoryList = [data[-1] for data in newDataSet]
    category = set(categoryList)
    if (len(category) == 1):
        flag = True
    return flag


def majorityCateg(newDataSet):
    categCount = {}
    categList = [data[-1] for data in newDataSet]
    for c in categList:
        if c not in categCount:
            categCount[c] = 1
        else:
            categCount[c] += 1
    sortedCateg = sorted(categCount.items(), key=lambda x: x[1], reverse=True)

    return sortedCateg[0][0]


# 创建ID3树
def createDecisionTreeID3(decisionTree, dataSet, labels):
    bestFeature = chooseBestFeatureID3(dataSet, labels)
    decisionTree[labels[bestFeature]] = {}
    currentLabel = labels[bestFeature]
    subSet = splitDataSetByFeature(bestFeature, dataSet)
    del (labels[bestFeature])
    newLabels = labels[:]
    for key in subSet:
        newDataSet = subSet[key]
        flag = checkIsOneCateg(newDataSet)
        if (flag == True):
            decisionTree[currentLabel][key] = newDataSet[0][-1]
        else:
            if (len(newDataSet[0]) == 1):  # 无特征值可划分
                decisionTree[currentLabel][key] = majorityCateg(newDataSet)
            else:
                decisionTree[currentLabel][key] = {}
                createDecisionTreeID3(decisionTree[currentLabel][key], newDataSet, newLabels)


# 创建C4.5树
def createDecisionTreeC45(decisionTree, dataSet, labels):
    bestFeature = chooseBestFeatureC45(dataSet, labels)
    decisionTree[labels[bestFeature]] = {}
    currentLabel = labels[bestFeature]
    subSet = splitDataSetByFeature(bestFeature, dataSet)
    del (labels[bestFeature])
    newLabels = labels[:]
    for key in subSet:
        newDataSet = subSet[key]
        flag = checkIsOneCateg(newDataSet)
        if (flag == True):
            decisionTree[currentLabel][key] = newDataSet[0][-1]
        else:
            if (len(newDataSet[0]) == 1):  # 无特征值可划分
                decisionTree[currentLabel][key] = majorityCateg(newDataSet)
            else:
                decisionTree[currentLabel][key] = {}
                createDecisionTreeC45(decisionTree[currentLabel][key], newDataSet, newLabels)


# 测试数据分类
def classifyTestData(decisionTree, testData):
    result1 = decisionTree['round'][testData[0]]
    if (type(result1) == str):
        category = result1
    else:
        category = decisionTree['round'][testData[0]]['red'][testData[1]]
    return category


if __name__ == '__main__':
    dataSetID3, labelsID3 = createDataSet()
    testData1 = [0, 1]
    testData2 = [1, 1]
    bestFeatureID3 = chooseBestFeatureID3(dataSetID3, labelsID3)
    decisionTreeID3 = {}
    createDecisionTreeID3(decisionTreeID3, dataSetID3, labelsID3)
    print("ID3 decision tree: ", decisionTreeID3)
    category1ID3 = classifyTestData(decisionTreeID3, testData1)
    print(testData1, ", classified as by ID3: ", category1ID3)
    category2ID3 = classifyTestData(decisionTreeID3, testData2)
    print(testData2, ", classified as by ID3: ", category2ID3)

    dataSetC45, labelsC45 = createDataSet()
    bestFeatureC45 = chooseBestFeatureC45(dataSetC45, labelsC45)
    decisionTreeC45 = {}
    createDecisionTreeC45(decisionTreeC45, dataSetC45, labelsC45)
    print("C4.5 decision tree: ", decisionTreeC45)
    category1C45 = classifyTestData(decisionTreeC45, testData1)
    print(testData1, ", classified as by C4.5: ", category1C45)
    category2C45 = classifyTestData(decisionTreeC45, testData2)
    print(testData2, ", classified as by C4.5: ", category2C45)
