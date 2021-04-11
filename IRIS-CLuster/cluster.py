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
from sklearn.metrics import calinski_harabaz_score
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


# 评估函数
def evaluate(label_true, label_pred, data):
    score_fmi = fowlkes_mallows_score(label_true, label_pred)
    score_ch = calinski_harabaz_score(data, label_pred)
    return score_fmi, score_ch

# K-Means 
def kmeans(data, target):
    # k-means聚类
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(data)
    # 得到聚类标签
    label_pred = kmeans_model.labels_
    # 评估
    score_fmi, score_ch = evaluate(target, label_pred, data)
    return label_pred, score_fmi, score_ch

# GMM 聚类
def gmm(data, target):
    gmm_model = GaussianMixture(n_components=3, random_state=42)
    gmm_model.fit(data)
    label_pred = gmm_model.predict(data)
    score_fmi, score_ch = evaluate(target, label_pred, data)
    return label_pred, score_fmi, score_ch

# 谱聚类
def spectral(data, target):
    spectral_model = SpectralClustering(n_clusters=3, random_state=42)
    spectral_model.fit(data)
    label_pred = spectral_model.labels_
    score_fmi, score_ch = evaluate(target, label_pred, data)
    return label_pred, score_fmi, score_ch


if __name__ == "__main__":
    # 加载数据
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    
    # 聚类
    kmeans_label_pred, kmeans_score_fmi, kmeans_score_ch = kmeans(data, target)
    gmm_label_pred, gmm_score_fmi, gmm_score_ch = gmm(data, target)
    spectral_label_pred, spectral_score_fmi, spectral_score_ch = spectral(data, target)

    # 打印评价指标
    print("\t\t FMI Score \t\t CH Score")
    print("K-Means \t", kmeans_score_fmi,"\t", kmeans_score_ch)
    print("GMM     \t", gmm_score_fmi, "\t", gmm_score_ch)
    print("Spectral\t", spectral_score_fmi, "\t", spectral_score_ch)

    # 可视化
    visualization(data, kmeans_label_pred, "K-Means Result")
    visualization(data, gmm_label_pred, "GMM Result")
    visualization(data, spectral_label_pred, "Spectral Result")