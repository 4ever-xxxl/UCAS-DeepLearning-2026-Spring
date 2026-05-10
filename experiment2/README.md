# 实验二 基于 ViT 的 CIFAR-10 图像分类

本实验使用 PyTorch 从零实现 Vision Transformer 相关模块，并在 CIFAR-10 数据集上完成 10 类图像分类任务。

## 实验目标

1. 理解并实现 ViT 中的 Patch Embedding、Multi-Head Self-Attention、Transformer Encoder 和 MLP 分类头。
2. 掌握图像分类任务中的数据读取、预处理、模型训练、验证和测试流程。
3. 在 CIFAR-10 测试集上达到 80% 以上分类准确率。

## 数据集

- 数据集：CIFAR-10
- 图像数量：60000 张彩色图像
- 图像尺寸：`3 x 32 x 32`
- 训练集：50000 张
- 测试集：10000 张
- 类别数：10 类，包括 airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck

## 模型结构

实验指导书要求基于 ViT 完成分类。当前实现保留 ViT 的核心结构，并针对 CIFAR-10 小尺寸图像做了轻量化改进：

`ConvPatchEmbed -> cls token + position embedding -> Transformer Encoder Blocks -> LayerNorm -> Linear classifier`

其中 Transformer Encoder Block 包含：

`LayerNorm -> Multi-Head Self-Attention -> residual -> LayerNorm -> MLP -> residual`

为了减少训练时间并提升小数据集泛化能力，模型采用 32x32 原生输入和卷积 tokenizer，不再将 CIFAR-10 图像放大到 224x224。

## 训练配置

- Batch size：512
- Epochs：最多 60，达到目标验证准确率后提前停止
- Optimizer：AdamW
- Learning rate：8e-4
- Weight decay：0.05
- Scheduler：5 epoch warmup + cosine annealing
- 数据增强：RandomCrop、RandomHorizontalFlip、RandAugment、RandomErasing
- 正则化：MixUp、label smoothing、dropout、stochastic depth
- 混合精度：CUDA 环境下启用 AMP

## 运行方式

训练并测试：

```bash
uv run python experiment2/train_vit.py
```

生成训练曲线、混淆矩阵、预测样本和特征图：

```bash
uv run python experiment2/generate_results.py
```

## 实验结果

当前结果来自 `experiment2/outputs/metrics.json`：

- 最佳验证集准确率：84.60%
- 测试集损失：0.7079
- 测试集准确率：83.36%

结果已达到实验要求的 80% 测试准确率。
