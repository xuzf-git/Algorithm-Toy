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
# feature_names = data.feature_names
# target_names = data.target_names
# print(data.target_names)

# 划分数据集
train_x, test_x, train_y, test_y = train_test_split(data.data,
                                                    data.target,
                                                    train_size=0.7)

# 构造 SVM
classifier = svm.SVC(kernel='linear')

# 训练 SVM
classifier.fit(train_x, train_y)

# 分类效果
print("train dataset: ", classifier.score(train_x, train_y))
print("test dataset: ", classifier.score(test_x, test_y))

# 预测
test_y_hat = classifier.predict(test_x)
train_y_hat = classifier.predict(train_x)
print("Accuracy on the test set: ", classifier.score(test_x, test_y))
print("test dataset predict result: ")
print(test_y_hat)

# 画图
feature_name = 'sepal length', 'sepal width', 'petal length', 'petal width'

plt.figure(figsize=(20, 10))
for i in range(0, 3, 2):
    plt.subplot(2, 2, i + 1)
    plt.scatter(train_x[:, i], train_x[:, i + 1], c=train_y.reshape((-1)), edgecolors='k', s=50)
    plt.xlabel(feature_name[i])
    plt.ylabel(feature_name[i + 1])
    plt.title("groundtruth")
    plt.subplot(2, 2, i + 2)
    plt.scatter(train_x[:, i], train_x[:, i + 1], c=train_y_hat.reshape((-1)), edgecolors='k', s=50)
    plt.xlabel(feature_name[i])
    plt.ylabel(feature_name[i + 1])
    plt.title("predict")
plt.show()