# import numpy as np
from numpy import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

X = array([[ 60.  ,   5.61],
       [110.  ,  10.81],
       [ 70.  ,   7.27],
       [ 90.  ,   9.05],
       [ 70.  ,   7.01],
       [ 60.  ,   6.32],
       [ 70.  ,   6.62]])
Y = array([[125.42],
       [432.15],
       [291.42],
       [ 94.54],
       [176.26],
       [ 48.11],
       [ 31.6 ]])

model_linear_a = LinearRegression().fit(X,Y)
print(model_linear_a.coef_)
print(model_linear_a.intercept_)
Y_pre1 = model_linear_a.predict(X)
print(Y_pre1)
print('===================')

learning_rate = 0.03
class RepeatLinearRegression(object):
    """
    重新构造多元线性回归类的方法，使用方法：传入X，Y矩阵      比对真实的原模型结果
    coef_ 为特征集X的系数  intercept_ 为截距
    解法2种情况：1，X.T*X 为满秩矩阵，直接求导；2，X.T*X 不为满秩矩阵，梯度下降法
    """
    def __init__(self):
        self.W = None
        self.coef_ =None
        self.m = None

    def stocGradAscent1(dataMatrix, resultLabels):
        m, n = shape(dataMatrix)
        alpha = 0.4
        weights = ones(n)  # initialize to all ones
        weightsHistory = zeros((40 * m, n))
        for j in range(40):
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                h = sum(dataMatrix[randIndex] * weights)
                error = resultLabels[randIndex] - h
                # print error
                weights = weights + alpha * error * dataMatrix[randIndex]
                weightsHistory[j * m + i, :] = weights
                del (dataIndex[randIndex])
        return weights

    def fit(self, X, Y):
        self.m = X.shape[0]
        wvalues1 = dot(X.T, X)
        # 判断是否是满秩矩阵
        if len(wvalues1[nonzero(wvalues1)]) >= wvalues1.shape[1]:
            wvalues2 = linalg.inv(wvalues1)
            wvalues3 = dot(dot(wvalues2, X.T), Y)
            self.coef_ = wvalues3
        else: # 梯度下降法求解
            self.coef_ = self.stocGradAscent1(X, Y)
        return self

    def predict(self, X):
        X = X.tolist()
        Y_pre = array([(self.coef_[0]*X[i][0] + self.coef_[1]*X[i][1]).tolist() for i in range(len(X))])
        return Y_pre

model_linear_test = RepeatLinearRegression().fit(X,Y)
print(model_linear_test.coef_)
Y_pre2 = model_linear_test.predict(X)
print(Y_pre2)

# 打印三维图表，样本点
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y[:, 0], label='parametric curve')
# 绘制LR() 的拟合线 ??待定
ax.plot(X[:, 0], X[:, 1], Y_pre1[:, 0], label='Y_pre1 curve')
ax.plot(X[:, 0], X[:, 1], Y_pre2[:, 0], label='Y_pre2 curve')
ax.legend()
plt.show()