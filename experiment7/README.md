# 实验 7：神经网络语言模型

本目录按照实验指导书要求，使用 PyTorch 在 Penn Treebank (PTB) 语料库上训练 LSTM 神经网络语言模型，并以困惑度（Perplexity, PPL）作为主要评价指标。

## 文件说明

- `train_ptb_lm.py`：端到端训练与测试脚本，包含 PTB 下载/加载、词表构建、BPTT 批生成、LSTM 模型、训练、验证、测试与 checkpoint 保存。
- `generate_results.py`：结果可视化脚本，生成训练曲线、next-word 预测样例和文本续写样例。
- `实验报告.md`：完整实验报告。
- `实验汇报PPT提纲.md`：10 页左右课堂汇报提纲。
- `data/`：PTB 数据缓存目录。
- `outputs/`：正式训练输出目录。
- `outputs_smoke/`：快速 smoke test 输出目录。

## 实验目标

1. 掌握循环神经网络和 LSTM 的基本结构。
2. 使用深度学习框架构建规范的多层 LSTM 语言模型。
3. 在 PTB 语料库上训练和评估语言模型。
4. 测试集困惑度目标：`PPL < 80`。

## 数据集

- **名称**：Penn Treebank (PTB) simple-examples
- **训练词数**：约 93 万 token
- **验证词数**：约 7.4 万 token
- **测试词数**：约 8.2 万 token
- **词表大小**：10,000
- **特殊标记**：`<eos>` 表示句末，`<unk>` 表示低频或未知词。

脚本优先使用指导书给出的 Mikolov `simple-examples.tgz` 地址；如果该压缩包下载不完整，则回退到 PTB 三个原始文本文件镜像，并进行 MD5 校验。

## 模型架构

默认 `large` preset 采用经典 PTB LSTM 配置：

`Embedding(10000, 1500) -> 2-layer LSTM(1500, dropout=0.65) -> Dropout -> Linear(1500, 10000)`

核心训练策略：

- BPTT 截断长度：35
- Batch size：20
- 优化器：SGD
- 初始学习率：1.0
- 梯度裁剪：global norm 10.0
- 学习率衰减：第 15 轮起每轮乘以 `1 / 1.15`
- CUDA 下启用 AMP、TF32 和 `torch.compile()`

## 运行方式

快速 smoke test：

```bash
uv run python experiment7/train_ptb_lm.py --preset test --device cpu --output-dir experiment7/outputs_smoke
```

正式训练：

```bash
uv run python experiment7/train_ptb_lm.py
```

常用参数：

```bash
uv run python experiment7/train_ptb_lm.py --preset medium
uv run python experiment7/train_ptb_lm.py --preset large --device cuda
uv run python experiment7/train_ptb_lm.py --preset large --disable-compile
```

生成可视化和样例：

```bash
uv run python experiment7/generate_results.py
```

## 输出内容

训练脚本生成：

- `experiment7/outputs/best_model.pt`：验证集 PPL 最低的模型 checkpoint。
- `experiment7/outputs/metrics.json`：训练历史、最佳验证 PPL、测试 PPL、next-word accuracy、是否达到 PPL<80。
- `experiment7/outputs/vocab.json`：PTB 词表。

可视化脚本生成：

- `experiment7/outputs/training_curves.png`：训练/验证 loss 与 PPL 曲线。
- `experiment7/outputs/next_word_examples.md`：测试集 next-word top-5 预测示例。
- `experiment7/outputs/generated_samples.md`：给定短语后的文本续写样例。

## 已验证结果

在当前环境（`torch 2.11.0+cu130`，GPU 为 `NVIDIA GeForce RTX 5070`）下，执行：

```bash
uv run python experiment7/train_ptb_lm.py --preset large --device cuda --output-dir experiment7/outputs
```

得到结果：

- 最佳验证集 loss：`4.4141`
- 最佳验证集困惑度：`82.61`
- 测试集 loss：`4.3687`
- 测试集困惑度：`78.94`
- 测试集 next-word accuracy：`27.86%`
- 模型参数量：`66,034,000`
- 达到指导书要求：测试集 `PPL < 80`

## 实验要求对应关系

- 构建规范 LSTM 网络：`PTBLanguageModel` 包含 Embedding、2 层 LSTM、Dropout、Softmax 分类层。
- 在 PTB 上训练和评估：`load_ptb_data`、`PTBBatchDataset`、`train_one_epoch`、`evaluate` 完成完整流程。
- 使用 PPL 评价：`metrics.json` 写入验证/测试困惑度。
- 目标 PPL<80：默认 `large` preset 的测试集 PPL 为 `78.94`。
