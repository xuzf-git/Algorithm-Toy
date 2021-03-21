'''
Description: 测试梯度下降法、最小二乘法的拟合效果
Author: xuzf
Date: 2021-03-21 08:54:51
FilePath: \algorithm-toy\Gradient-descent\evaluate.py
'''
from gd import grad_descent_fit
from ls import least_squares_fit
import matplotlib.pyplot as plt
import numpy as np


def main():
    x = [55, 71, 68, 87, 101, 87, 75, 78, 93, 73]
    y = [91, 101, 87, 109, 129, 98, 95, 101, 104, 93]
    w0_ls, w1_ls = least_squares_fit(x, y)
    w0_gd, w1_gd = grad_descent_fit(x, y, 500, 0.00001)
    print("最小二乘法：y = {} + {} * x".format(w0_ls, w1_ls))
    print("梯度下降法：y = {} + {} * x".format(w0_gd, w1_gd))
    x_plt = np.arange(50, 110)
    y_ls = w1_ls * x_plt + w0_ls
    y_gd = w1_gd * x_plt + w0_gd

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, '.')
    plt.plot(x_plt, y_ls, 'r--', label='least squares')
    plt.plot(x_plt, y_gd, 'g', label='grad descent')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()