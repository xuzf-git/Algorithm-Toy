<h1 style="text-align: center">Boston 房价预测</h1>

## 1 数据说明
| 序号 | 属性  | 含义                            | 序号 |  属性   | 含义                                |
| :--: | :---: | :------------------------------ | :--: | :-----: | :---------------------------------- |
|  1   | CRIM  | 城镇人均犯罪率                  |  8   |   DIS   | 距离 5 个波士顿的就业中心的加权距离 |
|  2   |  ZN   | 住宅用地所占比例                |  9   |   RAD   | 距离高速公路的便利指数              |
|  3   | INDUS | 城镇中非住宅用地所占比例        |  10  |   TAX   | 每一万美元的不动产税率              |
|  4   | CHAS  | 虚拟变量,用于回归分析           |  11  | PTRATIO | 城镇中的教师学生比例                |
|  5   |  NOX  | 环保指数                        |  12  |  BLACK  | 城镇中的黑人比例                    |
|  6   |  RM   | 每栋住宅的房间数                |  13  |  LSTAT  | 地区中有多少房东属于低收入人群      |
|  7   |  AGE  | 1940 年以前建成的自住单位的比例 |  14  |  MEDV   | 自住房屋房价中位数（也就是均价）    |

## 2 读取数据

1. 使用pandas.read_csv()函数读取CSV数据文件，保存到Pandas的DataFrame的数据结构中，通过DataFrame.describe()计算表示出各个数据的基本统计信息。如下图所示：
![](https://img.imgdb.cn/item/604a15135aedab222c2e035c.png)

2. 使用Matplotlib工具画出price-feature的散点图，进行数据可视化，以分析不同特征与房价的相关性。可视化结果如下图：
![](https://img.imgdb.cn/item/604a15b05aedab222c2ebf48.png)

## 3 特征选择

观察上图各个特征和房价的散点图，发现 'rm', 'lstat' 与价格的线性相关性最大，与 'age', 'dis' 特征的相关性也较高。因此应选择这四种特征作为模型预测房价的依据。

## 4 划分数据集
使用Sklearn库中的model_selection.train_test_split函数进行数据集划分。随机从数据集中选取30% 作为测试集，其余的70% 作为训练集，划分函数接口如下：

```
train_test_split(*arrays,test_size=None,train_size=None，random_state=None, shuffle=True, stratify=None)
1. *arrays: 第0维等长的可索引序(lists, numpy arrays, scipy-sparse matrices, pandas dataframes)
2. test_size: 若为float型，则表示测试集占比，若为int型，则表示测试集大小
3. train_size: 同test_size，控制训练集规模
4. random_state: 随机种子
5. shuffle：是否在划分前，将数据随机打乱
6. stratify：array型数据，将数据以stratify所谓类标签，分层划分。
```

## 5 构建线性回归模型

多元线性回归原理：线性回归是指预测结果与特征之间具有线性关系的一种预测模型。用来预测连续型变量。多元线性回归指的是多个特征的线性回归问题。对于一个有n个特征的样本i而言， $\hat y = \sum_i \theta_i \times x_i + bias$，构造该预测模型的关键是求解模型的参数。在Sklearn中的LinearRegerssion()使用最小二乘法进行参数估计。

调用`sklearn.linear_model.LinearRegression()` 构造线性回归模型，通过fit()函数，传入训练数据进行拟合。对于两种特征选择的分别进行训练。

参数如下：

| Feature | RM    | LSTAT  | AGE    | DIS    | Bias    |
| ------- | ----- | ------ | ------ | ------ | ------- |
| Weight  | 3.359 | -0.668 | -0.038 | -0.680 | 14.5981 |

使用`LinearRegression.predict()`在测试集上进行预测。

## 6 结果分析

误差

| Feature | RM+LSTAT+AGE+DIS | CRIM+RM+LSTAT |
| ------- | ---------------- | ------------- |
| mae     | 3.718085431      | 3.984112002   |
| mse     | 27.53756815      | 38.77070125   |

将上述模型的预测结果与真实值比较，结果如下：

横轴为真实值，纵轴为预测值，斜线表示真实值和预测值相同的坐标位置。

![](https://img.imgdb.cn/item/604a18eb5aedab222c324812.png)

观察发现，对于真实值较高的高价房屋区域，预测的误差较大，估价偏低。对于房价位于10~30范围内的房价预测的相对较为准确。