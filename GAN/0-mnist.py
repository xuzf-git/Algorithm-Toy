'''
Description: None
Author: xuzf
Date: 2021-05-08 23:02:21
FilePath: \algorithm-toy\GAN\0-mnist.py
'''

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Classier(nn.Module):
    def __init__(self):
        # 调用父类的构造函数，初始化父类
        super().__init__()
        # 定义神经网络
        self.model = nn.Sequential(
            nn.Linear(748, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
        # 创建损失函数
        self.loss_function = nn.MSELoss()
        # 创建优化器，随机梯度下降
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        # 训练过程监控
        self.counter = 0
        self.process = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        # 计算网络的输出值
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        # 反向传播
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        # 监控训练过程
        self.counter += 1
        if self.counter % 10== 0:
            self.process.append(loss.item())
            if self.counter % 10000 == 0:
                print('counter = {}'.format(self.counter))
    
    def plot_process(self):
        df = pd.DataFrame(self.process, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, ytricks=(0, 0.25, 0.5))


class MnistDataset(Dataset):
    def __init__(self, filePath):
        super().__init__()
        self.data_df = pd.read_csv(filePath, header=None)
        
    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, index):
        # 获取标签
        label = self.data_df.iloc[index, 0]
        # label 转 one-hot
        target = torch.zeros((10))
        target[label] = 1.0
        # 图像数据
        img = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0
        return label, img, target

    def plot_image(self, index):
        # 获取数据
        label = self.data_df.iloc[index, 0]
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)

        plt.title('label' + str(label))
        plt.imshow(img, interpolation='none', cmap='Blues')


def main():
    # 创建神经网络分类器
    classier = Classier()
    # 创建训练集
    mnist_dataset = MnistDataset('./data/mnist/mnist_train.csv')
    # 指定训练 epoch size
    epoch = 5
    for step in range(epoch):
        for label, img_data, target in mnist_dataset:
            classier.train(img_data, target)

if __name__ == '__main__':
    main()