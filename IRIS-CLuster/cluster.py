"""
Description: K-means 实现鸢尾花聚类
Author: xuzf
Date: 2021-04-10 15:22:02
FilePath: \\algorithm-toy\\IRIS-CLuster\\cluster.py
"""

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

from sklearn.manifold import TSNE
from sklearn.metrics import fowlkes_mallows_score
import matplotlib.pyplot as plt


# 数据降维可视化
def visualization(data, label_pred, title):
    tsne_data = TSNE(n_components=2, random_state=42).fit_transform(data)
    x0 = tsne_data[label_pred == 0]
    x1 = tsne_data[label_pred == 1]
    x2 = tsne_data[label_pred == 2]

    plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='*')
    plt.scatter(x1[:, 0], x1[:, 1], c='green', marker='o')
    plt.scatter(x2[:, 0], x2[:, 1], c='blue', marker='+')
    plt.title(title)
    plt.show()


# 评估函数：FMI评价法 FMI = TP / sqrt((TP + FP) * (TP + FN))
def evaluate(label_true, label_pred):
    score = fowlkes_mallows_score(
        label_true,
        label_pred,
    )
    return score


# K-Means
def kmeans(data, target):
    # k-means聚类
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(data)
    # 得到聚类标签
    label_pred = kmeans_model.labels_
    # 评估
    score = evaluate(target, label_pred)
    return label_pred, score


# GMM 聚类
def gmm(data, target):
    gmm_model = GaussianMixture(n_components=3, random_state=42)
    gmm_model.fit(data)
    label_pred = gmm_model.predict(data)
    score = evaluate(target, label_pred)
    return label_pred, score


# 谱聚类
def spectral(data, target):
    spectral_model = SpectralClustering(n_clusters=3, random_state=42)
    spectral_model.fit(data)
    label_pred = spectral_model.labels_
    score = evaluate(target, label_pred)
    return label_pred, score


# 加载数据
iris = datasets.load_iris()
data = iris.data
target = iris.target

kmeans_label_pred, kmeans_score = kmeans(data, target)
spectral_label_pred, spectral_score = spectral(data, target)
gmm_label_pred, gmm_score = gmm(data, target)

# 可视化
visualization(data, kmeans_label_pred, "K-Means Result")
visualization(data, spectral_label_pred, "Spectral Result")
visualization(data, gmm_label_pred, "GMM Result")

# 打印评价指标
print("K-Means Score:\t", kmeans_score)
print("Spectral Score:\t", spectral_score)
print("GMM Score:\t", gmm_score)
