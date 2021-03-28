'''
Description: 鸢尾花分类 (iris dataset)
Author: xuzf
Date: 2021-03-27 10:44:58
FilePath: \algorithm-toy\IRIS-SVM\iris-svm.py
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt

# 加载数据
data = load_iris()
print(data.target_names)

# data_x = data.data;
# data_y = data.target;

# 划分数据集
train_x, test_x, train_y, test_y = train_test_split(data.data,
                                                    data.target,
                                                    train_size=0.7)

# 构造 SVM
model = svm.SVC(kernel='linear')
model.fit(train_x, train_y)

feature_names = data.feature_names
target_names = data.target_names

# print("X's type", type(X))
# print("Y's type", type(Y))
# print("feature names's type", type(feature_names))
# print("target names's type", type(target_names))
# print(type(data))