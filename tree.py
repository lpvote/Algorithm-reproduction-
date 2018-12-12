# import numpy as np
from numpy import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import log
import operator

lense = open('lenses.txt')
lense_ = pd.DataFrame(columns =['young','myope','no','normal','soft','lenses'])
i = 0
for li in lense.readlines():
    i = i+1
    li_ = li.split()
    if len(li_) == 5:
        li_.append('null')
    lense_.loc[i,:] = li_


def prep(data_list):
    '''
    遍历数列构建成分类传出，快捷计算分类情况
    传入：计算分类情况的目标list   传出:分类及数量 的字典 {'A':12 ,'B':4}
    '''
    dict_count = {}
    for label in data_list:
        # 统计数据集中每种分类的个数
        if label not in dict_count.keys():
            dict_count[label] = 1
        else:
            dict_count[label] +=1
    return dict_count

def calc_entropy(data_set):
    '''
    data_set 传入数据集，整个或截片部分 ,最后一列是分类目标
    计算熵，需要求每个分类的概率值p_k和数量cou_k
    用字典的方式储存数值，数量方便求取,推荐用字典构建节点和子节点
    label 类标签 label_counts 各类数量   entropy 信息熵值
    '''
    count = len(data_set)
    label_counts = prep(data_set[:,-1])
    #计算熵 entropy
    entropy = 0.0
    for label in label_counts.keys():
        prob = float(label_counts[label]) / count
        entropy -= prob*log(prob, 2)
    return entropy

def splitDataSet(dataSet, axis, value):
    '''
    axis：特征的坐标。axis=0时,第0个特征其值可能为0或1,
    value=1时，dataSet前3个都符合，从而得到子集[[1,"yes"],[1,"yes"],[0,"no"]]。
    subDataSet  = splitDataSet(dataSet, i, value)        #对第i个特征，其值为value划分数据
    '''
    retDataSet = []             #需要新创建一个列表变量，因列表的参数是按照引用方式传递的
    for featVec in dataSet:
        if featVec[axis] == value:
            retDataSet.append(featVec.tolist())
    return array(retDataSet)


def cacl_gain(data_set):
    '''
    选取父节点：信息增益最高的特征
    需要计算的：每个特征的内部分类，prob = 特征集该分类/count 占比 ；
              calc_entropy(data_set_n--按某特征分类而切割的数据集)
    :param data_set:数据集
    :return:父节点
    '''
    count = len(data_set)
    base_entropy = calc_entropy(data_set) #总数据集熵传入总的数据集
    before_gain = 0.0
    for n in range(data_set.shape[1]):
        data_feature = data_set[:, n]
        data_feature_count = prep(data_set[:, n]) #传出的是各个feature 的字典，包括key 与values
        entropy = 0.0
        for label in data_feature_count.keys():
            subDataSet = splitDataSet(data_set, n, label)  #对第i个特征，其值为value划分数据
            prob = float(data_feature_count[label]) / count
            entropy -= prob * calc_entropy(subDataSet)
        after_gain = base_entropy - entropy
        if after_gain > before_gain:
            before_gain = after_gain
            choose_feature = n
    return choose_feature


def create_division_tree(data_set):
    """创建，选取了最佳特征后，为父节点。子节点为该特征的分类值划分，再去考虑其他特征的分类。
    划分真正的子节点，下一个特征。
    此时计算的信息增益就为 Gain(Dv,a) ,数据集为父节点来计算信息增益--中间包括父节点熵和子分支熵。
    所以，信息增益仍可以调用函数，递归函数
    """
    dict_tree = {}
    # 所有分类相同时返回
    if data_set.count(data_set[0]) == len(data_set):
        return data_set[0]

    #已经做挑选的特征做剔除处理再选分支
    for ref in data_set.shape[1]:
        choose_feature = cacl_gain(data_set)
        label = data_set[choose_feature][0]
        data_set = splitDataSet(data_set, i, label)
        choose_feature = cacl_gain(data_set)
        dict_tree[choose_feature] = create_division_tree(split_data_set(data_set, best_feat, value))

        def createTree(dataSet, labels):
            classList = [example[-1] for example in dataSet]
            if classList.count(classList[0]) == len(classList):
                return classList[0]  # stop splitting when all of the classes are equal
            if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
                return majorityCnt(classList)
            bestFeat = chooseBestFeatureToSplit(dataSet)
            bestFeatLabel = labels[bestFeat]
            myTree = {bestFeatLabel: {}}
            del (labels[bestFeat])
            featValues = [example[bestFeat] for example in dataSet]
            uniqueVals = set(featValues)
            for value in uniqueVals:
                subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
                myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
            return myTree

        def classify(inputTree, featLabels, testVec):
            firstStr = inputTree.keys()[0]
            secondDict = inputTree[firstStr]
            featIndex = featLabels.index(firstStr)
            key = testVec[featIndex]
            valueOfFeat = secondDict[key]
            if isinstance(valueOfFeat, dict):
                classLabel = classify(valueOfFeat, featLabels, testVec)
            else:
                classLabel = valueOfFeat
            return classLabel