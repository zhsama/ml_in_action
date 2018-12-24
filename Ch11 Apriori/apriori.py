#!/usr/bin/env python
# -·- coding: utf-8 -·-
# @Time    : 2018/12/20 16:18
# @Author  : zhcf1ess
# @Site    : 
# @File    : apriori.py
# @Software: PyCharm
from time import sleep
from votesmart import votesmart


def loadDataSet():
    """
    加载数据集
    Returns:
        数据集
    """
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    创建集合C1(候选项集的列表)
    对dataSet去重排序后放入list中，然后转换所有的元素为frozenset
    Args:
        dataSet: 输入数据集

    Returns:
        frozenset: 返回一个frozenset格式的list

    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 遍历所有的元素，如果不在 C1 出现过，就 append
                C1.append([item])
    # 对数组从小到大排序
    # print('sort 前=', C1)
    C1.sort()
    # frozenset 表示冻结的 set 集合，元素不可改变
    # 这里必须要使用frozenset而不是set类型，因为之后必须要将这些集合作为字典键值使用，使用frozenset可以实现这一点，而set却做不到。
    # print('sort 后=', C1)
    # print('frozenset=', map(frozenset, C1))
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    """
    计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
    Args:
        D: 输入数据集
        Ck: 候选项集列表
        minSupport: 最小支持度

    Returns:
        retList: 支持度大于 minSupport 的集合
        supportData: 候选项集支持度数据
    """
    # ssCnt临时存放选数据集Ck的频率. 例如: a->10, b->5, c->8
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # s.issubset(t) 测试是否s中的每一个元素都在t中
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))  # 数据集 D 的数量
    retList = []
    supportData = {}
    for key in ssCnt:
        # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        support = ssCnt[key] / numItems
        if support >= minSupport:
            # 在 retList 的首位插入元素，只存储支持度满足频繁项集的值
            retList.insert(0, key)
        # 存储所有的候选项（key）和对应的支持度（support）
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    '''
    输入频繁项集列表 Lk 与返回的元素个数 k，然后输出所有可能的候选项集 Ck
    以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}
    仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作
    Args:
        Lk: 频繁项集列表
        k: 返回的项集元素个数（若元素的前 k-2 相同，就进行合并）

    Returns:
        retList: 元素两两合并的数据集

    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[: k - 2]
            L2 = list(Lk[j])[: k - 2]
            # print('-----i=', i, k-2, Lk, Lk[i], list(Lk[i])[: k-2])
            # print('-----j=', j, k-2, Lk, Lk[j], list(Lk[j])[: k-2])
            L1.sort()
            L2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            # 若元素的前 k-2 相同，就进行合并
            if L1 == L2:
                # 求并集
                # print('union=', Lk[i] | Lk[j], Lk[i], Lk[j])
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    """
    找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
    Args:
        dataSet: 原始数据集
        minSupport: 支持度的阈值

    Returns:
        L: 频繁项集的全集
        supportData: 所有元素和支持度的全集

    """
    # C1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = createC1(dataSet)
    # print('C1: ', C1)
    # 对每一行进行 set 转换，然后存放到集合中
    D = list(map(set, dataSet))
    # print 'D=', D
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(D, C1, minSupport)
    # print("L1=", L1, "\n", "outcome: ", supportData)

    # L 加了一层 list, L 一共 2 层 list
    L = [L1]
    k = 2
    # 判断 L 的第 k-2 项的数据长度是否 > 0。
    # 第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。
    # L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
    while (len(L[k - 2]) > 0):
        # print('k=', k, L, L[k-2])
        # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        Ck = aprioriGen(L[k - 2], k)
        # print('Ck', Ck)

        Lk, supK = scanD(D, Ck, minSupport)  # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)
        if len(Lk) == 0:
            break
        # Lk 表示满足频繁子项的集合，L 元素在增加，例如:
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
        L.append(Lk)
        k += 1
        # print 'k=', k, len(L[k-2])
    return L, supportData


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    对两个元素的频繁项计算可信度（confidence）
    例如： {1,2}/{1} 或者 {1,2}/{2} 看是否满足条件
    Args:
        freqSet: 频繁项集中的元素，例如: frozenset([1, 3])
        H: 频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
        supportData: 所有元素的支持度的字典
        brl: 关联规则列表的空数组
        minConf: 最小可信度

    Returns:
        prunedH: 记录 可信度大于阈值的集合

    """
    # 记录可信度大于最小可信度（minConf）的集合
    prunedH = []
    for conseq in H:
        # 假设 freqSet = frozenset([1, 3]), H = [frozenset([1]), frozenset([3])]
        # 那么现在需要求出 frozenset([1]) -> frozenset([3]) 的可信度和 frozenset([3]) -> frozenset([1]) 的可信度
        # print('confData=', freqSet, H, conseq, freqSet-conseq)
        # 支持度定义: a->b = support(a|b)/support(a). 假设freqSet = frozenset([1, 3]), conseq = [frozenset([1])]
        # 那么frozenset([1])至frozenset([3])的可信度为:
        # support(a|b)/support(a) = supportData[freqSet]/supportData[freqSet-conseq]
        # = supportData[frozenset([1,3])]/supportData[frozenset([1])]
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            # 只要买了 freqSet-conseq 集合，一定会买 conseq 集合（freqSet-conseq 集合和 conseq集合 是全集）
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    递归计算频繁项集的规则
    Args:
        freqSet: 频繁项集中的元素，例如: frozenset([2, 3, 5])
        H: 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
        supportData: 所有元素的支持度的字典
        brl: 关联规则列表的数组
        minConf: 最小可信度

    Returns:

    """
    """
    * H[0]是freqSet的元素组合的第一个元素,并且H中所有元素的长度都一样长度由aprioriGen(H,m+1)这里的m+1来控制
    * 该函数递归时，H[0] 的长度从 1 开始增长 1 2 3 ...
    * 假设 freqSet = frozenset([2, 3, 5]), H = [frozenset([2]), frozenset([3]), frozenset([5])]
    * 那么 m = len(H[0]) 的递归的值依次为 1 2* 
    * 在 m = 2 时, 跳出该递归。假设再递归一次，那么 H[0] = frozenset([2, 3, 5])，freqSet = frozenset([2, 3, 5])
    * 没必要再计算 freqSet 与 H[0] 的关联规则了。
    """
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        # print 'freqSet******************', len(freqSet), m + 1, freqSet, H, H[0]
        # 生成 m+1 个长度的所有可能的 H 中的组合，假设 H = [frozenset([2]), frozenset([3]), frozenset([5])]
        # 第一次递归调用时生成 [frozenset([2, 3]), frozenset([2, 5]), frozenset([3, 5])]
        # 第二次 。。。没有第二次，递归条件判断时已经退出了
        Hmp1 = aprioriGen(H, m + 1)
        # 返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print('Hmp1=', Hmp1)
        print('len(Hmp1)=', len(Hmp1), 'len(freqSet)=', len(freqSet))
        # 计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if (len(Hmp1) > 1):
            # print('----------------------', Hmp1)
            # print(len(freqSet),  len(Hmp1[0]) + 1)
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
    """
    生成关联规则
    Args:
        L: 频繁项集列表
        supportData: 频繁项集支持度的字典
        minConf: 最小置信度

    Returns:
        bigRuleList: 可信度规则列表（关于 (A->B+置信度) 3个字段的组合）

    """
    bigRuleList = []
    # 假设 L = [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])], [frozenset([1, 3]), frozenset([2, 5]),
    # frozenset([2, 3]), frozenset([3, 5])], [frozenset([2, 3, 5])]]
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        for freqSet in L[i]:
            # 假设：freqSet= frozenset([1, 3]), H1=[frozenset([1]), frozenset([3])]
            # 组合总的元素并遍历子元素，并转化为 frozenset 集合，再存放到 list 列表中
            H1 = [frozenset([item]) for item in freqSet]
            # 2 个的组合，走 else, 2 个以上的组合，走 if
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def getActionIds():
    # votesmart.apikey = 'get your api key first'
    votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
    actionIdList = []
    billTitleList = []
    fr = open('../data/ch11/recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)  # api call
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)  # delay to be polite
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList):  # this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']  # list of what each item stands for
    for billTitle in billTitleList:  # fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}  # list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if vote.candidateName not in transDict:
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning


def testApriori():
    # 加载测试数据集
    dataSet = loadDataSet()
    print('dataSet: ', dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.7)
    print('L(0.7): ', L1)
    print('supportData(0.7): ', supportData1)

    # Apriori 算法生成频繁项集以及它们的支持度
    L2, supportData2 = apriori(dataSet, minSupport=0.5)
    print('L(0.5): ', L2)
    print('supportData(0.5): ', supportData2)


def testGenerateRules():
    # 加载测试数据集
    dataSet = loadDataSet()
    print('dataSet: ', dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.5)
    print('L(0.7): ', L1)
    print('supportData(0.7): ', supportData1)

    # 生成关联规则
    rules = generateRules(L1, supportData1, minConf=0.5)
    print('rules: ', rules)


def congressVote():
    """
    构建美国国会投票记录的事务数据集
    Returns:

    """
    # 项目案例
    actionIdList, billTitleList = getActionIds()
    # 测试前2个
    # transDict, itemMeaning = getTransList(actionIdList[: 2], billTitleList[: 2])
    # transDict 表示 action_id的集合，transDict[key]这个就是action_id对应的选项，例如 [1, 2, 3]
    transDict, itemMeaning = getTransList(actionIdList, billTitleList)
    # 得到全集的数据
    dataSet = [transDict[key] for key in transDict.keys()]
    L, supportData = apriori(dataSet, minSupport=0.3)
    rules = generateRules(L, supportData, minConf=0.95)
    print(rules)


def deathCup():
    """
    发现毒蘑菇的相似特性
    Returns:

    """
    # 得到全集的数据
    dataSet = [line.split() for line in open("../data/ch11/mushroom.dat").readlines()]
    L, supportData = apriori(dataSet, minSupport=0.3)
    # 2表示毒蘑菇，1表示可食用的蘑菇
    # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
    for item in L[1]:
        if item.intersection('2'):
            print(item)

    for item in L[2]:
        if item.intersection('2'):
            print(item)


if __name__ == '__main__':
    # 测试 Apriori 算法
    testApriori()

    # 生成关联规则
    testGenerateRules()

    # 项目案例
    # 构建美国国会投票记录的事务数据集
    # congressVote()

    # 发现毒蘑菇的相似特性
    # deathCup()
    pass
