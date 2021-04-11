# IRIS-Cluster 算法分析

对于没有类别标签的数据点，聚类算法分析数据的特征将样本集划分成若干个不相交的子集。本实验分析比较K-Means、使用GMM模型的EM聚类、谱聚类三种算法。

## 1 K-Means 算法
​		K-Means算法在给定聚类个数K之后，随机选取K个数据点作为每个簇的聚类中心，根据定义的数据点之间的距离确定每个点所属的簇。计算每个簇的均值数据点，将其作为新的聚类中心，重复以上过程，直到每个簇中心不再变化。
​		K-Means算法的迭代过程实际上是以最小化平方误差为目标的EM算法。而以均方误差为优化目标，限制了聚类的边界为圆形（高维球体）。在很多情况下（如条形簇等），圆形的边界并不能很好地划分。

```python
from sklearn.cluster import Kmeans
# k-means聚类
kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(data)
# 得到聚类标签
label_pred = kmeans_model.labels_
```
## 2 使用GMM模型的EM聚类
​		GMM模型是高斯分布的线性组合，每一个高斯分量用均值和标准差描述一个簇。对于高维特征，分模型在每个特征维度都有均值和标准差描述，因此GMM描述的簇的边界为任意形状的椭圆形（高维椭球）。  

​		与K-Means相同，GMM的训练过程同样使用EM算法。选择簇的数量K并随机初始化每个簇的高斯分布参数（均值和方差）。给定每个簇的高斯分布，计算每个数据点属于每个簇的概率。基于这些概率我们计算高斯分布参数使得数据点的概率最大化，可以使用数据点概率的加权来计算这些新的参数，权重就是数据点属于该簇的概率。
​		GMM模型与K-Means的区别还在于在判断一个数据点属于哪个簇时，K-Means认为该点属于离它最近的簇，属于硬分类，而GMM给出该点属于每个簇的概率，属于软分类。

```python
from sklearn.mixture import GaussianMixture
# GMM聚类
gmm_model = GaussianMixture(n_components=3, random_state=42)
gmm_model.fit(data)
label_pred = gmm_model.predict(data)
```
## 3 谱聚类
​		谱聚类主要思想是把所有的数据看做空间中的点，这些点之间可以用边连接起来。距离较远的两个点之间的边权重值较低，而距离较近的两个点之间的边权重值较高，通过对所有数据点组成的图进行切图，让切图后不同的子图间边权重和尽可能的低，而子图内的边权重和尽可能的高，从而达到聚类的目的。
​算法流程:

* 使用高斯核函数计算点之间的相似度生成亲和矩阵，构造图。
* 计算图的拉普拉斯矩阵。
* 求解矩阵的前K小的特征值对应的特征向量，将以上特征向量标准化并组成 n×k 的矩阵
* 使用聚类算法（如K-Means）对矩阵中的每一行进行聚类。
  使用sklearn.mixture.GaussianMixture作为GMM的实现。
```python
from sklearn.cluster import SpectralClustering
# 谱聚类
spectral_model = SpectralClustering(n_clusters=3, random_state=42)
spectral_model.fit(data)
label_pred = spectral_model.labels_
```
## 4 实验结果及分析

聚类结果如下图所示 

评价采用FMI和CH指标。
FMI 计算公式如下：
$$
FMI=\frac{TP}{\sqrt{(TP+FP)*(TP+FN)}}
$$
CH 指标计算公式如下：
$$
CH=\frac{tr(B)/(K-1))}{tr(W)/(N-K)}
$$
其中:$tr(b)=\sum||z_j-z||^2 $表示簇间距离差矩阵的迹、 $tr(W)=\sum\sum||x_i-z_j ||^2$ 表示簇内离差矩阵的迹、$z$ 表示数据集的均值、$z_j$  表示第  $j$ 个簇的均值	

|          | FMI Score | CH Score |
| :------: | :-------: | :------: |
| K-Means  |  0.8208   |  560.4   |
|   GMM    |  0.9356   |  480.8   |
| Spectral |  0.8294   |  558.9   |


* FMI反映了聚类结果与标签的匹配程度，比较结果发现GMM与标签匹配程度最佳，K-Means最差。
* CH系数表示聚类结果的簇间距离与簇内聚类之比，体现了利用数据内部特征评价的聚类性能，比较结果发现K-Means与谱聚类效果最优，GMM反而最差。

​   观察降维可视化的图像，分析以上结果：GMM的聚类边界明显不同于K-Means和谱聚类，而K-Means和谱聚类的聚类边界相似。由于谱聚类能收敛到全局最优解，因此可以认为在当前数据集上的该边界是最好的。但是观察GMM的聚类结果可以发现：**谱聚类和K-Means的聚类结果与标签出现的较大偏差是由于GMM图像中最上方的异常值导致的。**

​		分析K-Means和谱聚类的特点，这两种算法判断数据点是否属于某簇时是使用硬分类，且均使用了平均值，因此对于噪声都很敏感，相反GMM采用了软分类，对噪声相对不敏感，因此GMM获得最好的聚类效果。