# 实验一 手写数字识别汇报提纲

## 第 1 页：实验目标

- 掌握卷积神经网络基本原理
- 熟悉 PyTorch 搭建与训练流程
- 在 MNIST 上实现手写数字识别并达到 98% 以上准确率

## 第 2 页：实验环境与数据集

- 框架：PyTorch 2.11.0+cu130
- 硬件：NVIDIA GeForce RTX 5070
- 数据集：MNIST
- 输入尺寸：`1 x 28 x 28`
- 分类类别：10 类数字

## 第 3 页：模型结构

- `Conv2d(1, 32) -> BN -> ReLU`
- `Conv2d(32, 32) -> BN -> ReLU -> MaxPool -> Dropout`
- `Conv2d(32, 64) -> BN -> ReLU`
- `Conv2d(64, 64) -> BN -> ReLU -> MaxPool -> Dropout`
- `Flatten -> Linear(3136, 128) -> ReLU -> Dropout -> Linear(128, 10)`

## 第 4 页：训练配置

- Batch size：128
- Epochs：8
- Optimizer：Adam
- Loss：CrossEntropyLoss
- 数据增强：RandomAffine
- 验证策略：从训练集划分 5000 张作为验证集，保存最佳模型

## 第 5 页：实验结果

- 最佳验证集准确率：99.36%
- 测试集损失：0.0157
- 测试集准确率：99.51%
- 结果满足并超过实验要求

## 第 6 页：结论与改进方向

- 已完成基于 CNN 的 MNIST 手写数字识别实验
- 模型在标准数据集上表现稳定，泛化能力较好
- 后续可尝试：
  - 更深层网络结构
  - 学习率调度策略
  - 可视化混淆矩阵与错误样本分析
