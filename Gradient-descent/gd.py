'''
Description: 使用梯度下降法求解一元线性回归方程 y(x,w) = w0 + w1 * x 系数
Author: xuzf
Date: 2021-03-21 07:55:04
FilePath: \algorithm-toy\Gradient-descent\gd.py
'''
import numpy as np

def grad_descent_fit(X, Y, steps, lr):
    X_array = np.array(X)
    Y_array = np.array(Y)
    n = len(X)
    # 随机初始化参数
    w0 = np.random.random()
    w1 = np.random.random();

    for i in range(steps):
        # 计算梯度
        Loss_array = w1 * X_array + w0 - Y_array
        grad_w0 = 2 * np.sum(Loss_array) / n
        grad_w1 = 2 * np.dot(Loss_array, X_array) / n
        # 更新参数
        w0 = w0 - lr * grad_w0
        w1 = w1 - lr * grad_w1
    return w0, w1

