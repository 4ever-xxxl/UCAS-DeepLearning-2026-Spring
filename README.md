# Deep Learning Experiments (2026 Spring)

本项目是 2026 年春季学期《深度学习》课程的实验代码仓库。包含多个基于 PyTorch 的深度学习实验，涵盖了计算机视觉（CV）和自然语言处理（NLP）的核心任务。

## 实验概览

| 实验编号 | 任务名称 | 核心技术 | 数据集 | 关键指标 |
| :--- | :--- | :--- | :--- | :--- |
| **Experiment 1** | [手写数字识别](./experiment1) | CNN, PyTorch | MNIST | Test Acc: 99.51% |
| **Experiment 2** | [图像分类](./experiment2) | ViT (Vision Transformer) | CIFAR-10 | Test Acc: 83.36% |
| **Experiment 3** | [自动写诗](./experiment3) | LSTM, Char-level LM | Tang Poetry | Perplexity (PPL) |
| **Experiment 4** | [神经机器翻译](./experiment4) | Transformer (Encoder-Decoder) | NiuTrans (Zh-En) | BLEU-4 ≥ 14 |
| **Experiment 7** | [神经网络语言模型](./experiment7) | Multi-layer LSTM | PTB | Test PPL: 78.94 |

## 环境准备

本项目使用 [uv](https://github.com/astral-sh/uv) 管理 Python 虚拟环境和依赖。

### 1. 安装 uv
参考 [uv 官方文档](https://github.com/astral-sh/uv#installation)进行安装。

### 2. 初始化环境
在项目根目录下执行以下命令，将自动创建虚拟环境并安装所有依赖：

```bash
uv sync
```

## 运行实验

每个实验目录都包含独立的训练脚本和结果生成脚本。

### 训练模型
以实验 1 为例：
```bash
uv run python experiment1/train_mnist.py
```

### 生成结果与可视化
训练完成后，可以运行对应的 `generate_results.py` 生成图表和评估报告：
```bash
uv run python experiment1/generate_results.py
```

## 项目结构

```text
.
├── docs/               # 课程相关文档与资料
├── experiment1/        # 实验 1: 手写数字识别 (CNN)
├── experiment2/        # 实验 2: 图像分类 (ViT)
├── experiment3/        # 实验 3: 自动写诗 (LSTM)
├── experiment4/        # 实验 4: 机器翻译 (Transformer)
├── experiment7/        # 实验 7: 语言模型 (PTB LSTM)
├── pyproject.toml      # 项目配置文件 (uv)
└── README.md           # 本文件
```

## 许可证

本项目采用 [MIT License](LICENSE) 协议。
