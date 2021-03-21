'''
Description: 使用最小二乘法求解一元线性回归方程 y(x,w) = w0 + w1 * x 系数
Author: xuzf
Date: 2021-03-21 07:56:48
FilePath: \algorithm-toy\Gradient-descent\Least-squares-method.py
'''
import numpy as np


def least_squares_fit(X, Y):
    X_array = np.array(X)
    Y_array = np.array(Y)
    n = len(X)

    w1 = n * np.dot(X_array, Y_array) - np.sum(X_array) * np.sum(Y_array)
    w1 = w1 / (n * np.dot(X_array, X_array) - np.sum(X_array)**2)
    w0 = (np.sum(Y_array) - w1 * np.sum(X_array)) / n

    return w0, w1