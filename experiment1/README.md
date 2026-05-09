# 实验 1：手写数字识别

本目录按照实验指导书要求，使用 PyTorch 在 MNIST 数据集上完成卷积神经网络的训练与评估。

## 文件说明

- `train_mnist.py`：端到端训练与测试脚本，包含数据加载、模型定义、训练、验证、测试、模型保存。
- `实验报告.md`：实验报告模板与结果记录。
- `实验1+手写数字识别实验指导书.pdf`：原始实验指导书。

## 环境

项目依赖已经在根目录 `pyproject.toml` 中声明：

- `torch`
- `torchvision`

## 运行方式

在项目根目录执行：

```bash
uv run python experiment1/train_mnist.py
```

常用参数：

```bash
uv run python experiment1/train_mnist.py --epochs 8 --batch-size 128
uv run python experiment1/train_mnist.py --device cpu
uv run python experiment1/train_mnist.py --disable-augment
```

## 输出内容

脚本默认会生成：

- `experiment1/outputs/best_model.pt`：验证集表现最好的模型权重。
- `experiment1/outputs/metrics.json`：训练过程和测试指标。
- `experiment1/data/`：MNIST 数据缓存目录。

## 已验证结果

在当前环境（`torch 2.11.0+cu130`，`torchvision 0.26.0+cu130`，GPU 为 `NVIDIA GeForce RTX 5070`）下，执行默认命令：

```bash
uv run python experiment1/train_mnist.py
```

得到结果：

- 最佳验证集准确率：`99.36%`
- 测试集损失：`0.0157`
- 测试集准确率：`99.51%`

## 实验要求对应关系

- 搭建 PyTorch 环境：通过项目依赖与 `uv` 环境完成。
- 构建规范卷积神经网络：`MNISTCNN` 包含卷积、激活、池化、展平、全连接层。
- 在 MNIST 上训练并评估：`train_mnist.py` 完成训练、验证和测试流程。
- 测试集准确率达到 98% 及以上：默认配置面向该目标设计，最终结果会写入 `metrics.json`。
