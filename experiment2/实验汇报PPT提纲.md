# 实验二 基于 ViT 的 CIFAR10 图像分类汇报提纲

## 第 1 页：实验目标

- 使用 PyTorch 从零实现 ViT 图像分类模型
- 理解 Attention、Transformer Encoder 和 MLP 分类头的作用
- 在 CIFAR-10 测试集上达到 80% 以上准确率

## 第 2 页：实验环境与数据集

- 框架：PyTorch 2.11.0+cu130、torchvision 0.26.0+cu130
- 硬件：NVIDIA GeForce RTX 5070
- 数据集：CIFAR-10
- 输入尺寸：`3 x 32 x 32`
- 数据规模：训练集 50000 张，测试集 10000 张
- 分类类别：10 类

## 第 3 页：模型结构

- 输入图像：CIFAR-10 RGB 图像
- 特征嵌入：卷积 tokenizer 直接处理 32x32 输入
- 序列建模：加入 cls token 和 position embedding
- 编码器：6 层 Transformer Encoder Block
- 注意力：Multi-Head Self-Attention，8 个 head
- 分类头：LayerNorm 后接 Linear 输出 10 类概率

## 第 4 页：训练配置

- Batch size：512
- 最大 Epochs：60，达到目标后提前停止
- Optimizer：AdamW
- Learning rate：8e-4
- Scheduler：warmup + cosine annealing
- 数据增强：RandomCrop、RandomHorizontalFlip、RandAugment、RandomErasing
- 正则化：MixUp、label smoothing、dropout、stochastic depth

## 第 5 页：实验结果

- 最佳验证集准确率：84.60%
- 测试集损失：0.7079
- 测试集准确率：83.36%
- 指标要求：测试集准确率 80% 以上
- 可视化结果：
  - 训练曲线：`outputs/training_curves.png`
  - 混淆矩阵：`outputs/confusion_matrix.png`
  - 预测样本：`outputs/predictions.png`
  - 特征图：`outputs/feature_maps.png`

## 第 6 页：结论与改进方向

- 已完成基于 ViT 的 CIFAR-10 图像分类实验
- 测试集准确率达到 83.36%，超过实验要求
- 卷积 tokenizer 降低训练时间并改善小数据集泛化
- 后续可尝试：
  - 更长训练并调低学习率末期衰减
  - CutMix 与 MixUp 联合策略
  - 使用预训练 ViT 或蒸馏方法进一步提升准确率
