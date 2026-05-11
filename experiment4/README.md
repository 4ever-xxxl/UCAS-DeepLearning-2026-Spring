# 实验四 基于 Transformer 的神经机器翻译

本实验使用 PyTorch 从零实现 Transformer 模型，在 NiuTrans 中英平行语料库上完成中文→英文神经机器翻译任务。

## 实验目标

1. 理解并实现 Transformer 的 Encoder-Decoder 架构，包括 Multi-Head Self-Attention、Cross-Attention、Positional Encoding 等核心组件。
2. 掌握 NMT 任务中的数据预处理、词汇表构建、Teacher Forcing 训练、Beam Search 解码等流程。
3. 在 NiuTrans 测试集上达到 BLEU-4 ≥ 14 的翻译质量。

## 数据集

- 数据集：NiuTrans 开源中英平行语料库
- 训练集：100,000 句中英平行句对
- 验证集：600 句（Dev-set）
- 测试集：1,000 句中文，每句配有 3 个英文参考译文
- 中文已分词（空格分隔），英文已完成 tokenization 和小写化

## 模型结构

基于 "Attention Is All You Need" 的 Transformer 架构：

```
Input → Embedding + Positional Encoding
     → Encoder (N layers)
       → Self-Attention → Feed-Forward (with residual + LayerNorm)
     → Decoder (N layers)
       → Masked Self-Attention → Cross-Attention → Feed-Forward
     → Linear Projection → Softmax → Output
```

针对 10 万句对的较小规模数据，采用轻量化配置：d_model=256, heads=8, layers=4。

## 训练配置

- Batch size：64（动态填充）
- Epochs：30
- Optimizer：Adam（β₁=0.9, β₂=0.98, ε=10⁻⁹）
- Learning rate：Transformer 论文 warmup 策略（warmup_steps=4000）
- 正则化：Dropout=0.2, Label Smoothing=0.1
- 混合精度：CUDA 环境下启用 AMP

## 运行方式

训练并测试：

```bash
uv run python experiment4/train_nmt.py
```

生成可视化结果：

```bash
uv run python experiment4/generate_results.py
```
