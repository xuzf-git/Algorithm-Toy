'''
Description: 生成1010格式数据的GAN
Author: xuzf
Date: 2021-05-09 14:22:22
FilePath: /Algorithm-Toy/GAN/1-1010-GAN.py
'''

import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_real():
    '''
    @description: 生成1010格式的训练正样本(引入高斯噪声)
    @param {none}
    @return {tensor}
    '''
    real_data = torch.FloatTensor([
        random.uniform(0.8, 1.0),
        random.uniform(0.0, 0.2),
        random.uniform(0.8, 1.0),
        random.uniform(0.0, 0.2),
    ])
    return real_data


def generate_random(size):
    '''
    @description: 生成训练负样本
    @param {size: 生成样本的维度}
    @return {tensor}
    '''
    random_data = torch.rand(size)
    return random_data


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        # 调用父类的构造函数，初始化父类
        super().__init__()
        # 定义神经网络
        self.model = nn.Sequential(nn.Linear(4, 3), nn.Sigmoid(),
                                   nn.Linear(3, 1), nn.Sigmoid())
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
        if self.counter % 10 == 0:
            self.process.append(loss.item())
            if self.counter % 10000 == 0:
                print('counter = {}'.format(self.counter))

    def plot_process(self):
        df = pd.DataFrame(self.process, columns=['loss'])
        df.plot(ylim=(0, 1.0),
                figsize=(16, 8),
                alpha=0.1,
                marker='.',
                grid=True)


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义神经网络
        self.model = nn.Sequential(nn.Linear(1, 3), nn.Sigmoid(),
                                   nn.Linear(3, 4), nn.Sigmoid())
        # 创建优化器，随机梯度下降
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        # 训练过程监控
        self.counter = 0
        self.process = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, discriminator, inputs, targets):
        # 生成器输出
        gen_data = self.forward(inputs)
        # 判别器预测
        pred_data = discriminator(gen_data)
        # 计算损失
        self.optimiser.zero_grad()
        loss = discriminator.loss_function(pred_data, targets)
        # 监控训练过程
        self.counter += 1
        if self.counter % 10 == 0:
            self.process.append(loss.item())
        # 从判别器开始误差梯度的反向传播
        loss.backward()
        # 用生成器的优化器更新自身参数
        self.optimiser.step()

    def plot_process(self):
        df = pd.DataFrame(self.process, columns=['loss'])
        df.plot(ylim=(0, 1.0),
                figsize=(16, 8),
                alpha=0.1,
                marker='.',
                grid=True)


# 创建判别器和生成器
discriminator = Discriminator()
generator = Generator()

gen_process = []
for i in range(10000):
    # 用真实样本训练判别器
    discriminator.train(generate_real(), torch.FloatTensor([1.0]))
    # 用生成样本训练判别器
    discriminator.train(
        generator(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
    # 训练生成器
    generator.train(discriminator, torch.FloatTensor([0.5]),
                    torch.FloatTensor([1.0]))
    if i % 1000 == 0:
        gen_process.append(
            generator(torch.FloatTensor([0.5])).detach().numpy())

# 可视化训练过程中的判别器损失
discriminator.plot_process()
# 可视化训练过程中的生成器损失
generator.plot_process()
# 可视化训练过程中的生成效果
plt.figure(figsize=(16, 8))
plt.imshow(np.array(gen_process).T, interpolation='none', cmap='Blues')
