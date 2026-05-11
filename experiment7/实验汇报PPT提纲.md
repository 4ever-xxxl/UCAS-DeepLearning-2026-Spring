# 实验7 神经网络语言模型汇报提纲

## 第 1 页：封面

- 实验7：神经网络语言模型
- Penn Treebank 上的 LSTM 语言模型
- 深度学习课程实验
- 2026 年 5 月

## 第 2 页：目录

1. 实验任务概述
2. 数据集介绍
3. 模型结构设计
4. 损失函数与评价指标
5. 优化器与训练策略
6. 工程实现与创新点
7. 训练配置
8. 实验结果
9. 结果分析
10. 总结与改进方向

## 第 3 页：实验任务概述

- **任务定义**：词级语言模型，给定前文词序列预测下一个词。
- **应用场景**：机器翻译重排序、语音识别、文本生成、语言理解预训练。
- **核心目标**：在 PTB 测试集上达到 `PPL < 80`。
- **技术路线**：Embedding + 2 层 LSTM + Dropout + Softmax。
- **训练方式**：截断 BPTT，按文本原始顺序保持跨 batch 的隐藏状态。

## 第 4 页：数据集介绍

- **数据集**：Penn Treebank (PTB) simple-examples
- **词表大小**：10,000
- **训练集**：约 93 万 token
- **验证集**：约 7.4 万 token
- **测试集**：约 8.2 万 token
- **特殊 token**：
  - `<eos>`：句末标记
  - `<unk>`：低频词/未知词
- **预处理**：换行替换为 `<eos>`，训练集建词表，valid/test 映射到同一词表。

## 第 5 页：模型结构设计

默认 large preset：

| 模块 | 配置 |
|---|---|
| Embedding | 10000 -> 1500 |
| LSTM | 2 层，hidden=1500 |
| Dropout | 0.65 |
| Classifier | Linear 1500 -> 10000 |
| 参数量 | 66,034,000 |

结构流程：

`word ids -> embedding -> dropout -> LSTM x2 -> dropout -> linear -> next-word logits`

设计理由：

- LSTM 适合顺序文本建模，能够通过门控机制保留较长上下文。
- 大 hidden size 提高模型容量。
- 强 dropout 适应 PTB 小语料，降低过拟合风险。

## 第 6 页：损失函数与评价指标

- **损失函数**：交叉熵损失 `CrossEntropyLoss`
- **训练目标**：最大化真实下一个词的条件概率
- **困惑度定义**：`PPL = exp(cross_entropy_loss)`
- **指标解释**：
  - PPL 越低，语言模型越好。
  - 实验要求测试集 PPL<80。
  - next-word accuracy 作为辅助指标，不作为主要达标依据。

关键代码：

```python
logits, hidden = model(inputs, hidden)
loss = criterion(logits, targets.reshape(-1))
perplexity = exp(loss)
```

## 第 7 页：优化器与训练策略

- **优化器**：SGD
- **初始学习率**：1.0
- **学习率衰减**：第 15 轮起，每轮乘以 `1 / 1.15`
- **梯度裁剪**：global norm 10.0
- **BPTT 长度**：35
- **Batch size**：20
- **Epochs**：55
- **硬件优化**：
  - AMP 混合精度
  - TF32
  - `torch.compile(mode="reduce-overhead")`

## 第 8 页：工程实现与创新点

- PyTorch 复现实验指导书中的 TensorFlow PTB 流程。
- `PTBBatchDataset` 对齐原始 `ptb_producer` 思路：连续文本流、固定 batch、目标右移一位。
- 提供 `test/small/medium/large` preset：
  - `test`：快速验证代码链路
  - `large`：正式达标训练
- 下载鲁棒性：
  - 首选 Mikolov simple-examples.tgz
  - 失败时回退到 PTB 原始三文件镜像并进行 MD5 校验
- 输出完整：
  - best checkpoint
  - metrics.json
  - 训练曲线
  - next-word top-5 预测样例
  - 续写样例

## 第 9 页：训练配置

正式训练命令：

```bash
uv run python experiment7/train_ptb_lm.py --preset large --device cuda --output-dir experiment7/outputs
```

配置表：

| 项目 | 值 |
|---|---:|
| Preset | large |
| Epochs | 55 |
| Batch size | 20 |
| BPTT steps | 35 |
| Embedding dim | 1500 |
| Hidden dim | 1500 |
| LSTM layers | 2 |
| Dropout | 0.65 |
| Optimizer | SGD |
| Initial LR | 1.0 |
| Grad clip | 10.0 |
| 训练硬件 | RTX 5070 12GB |

## 第 10 页：实验结果

结果来自 `experiment7/outputs/metrics.json`：

| 指标 | 数值 |
|---|---:|
| 最佳验证集 loss | 4.4141 |
| 最佳验证集 PPL | 82.61 |
| 测试集 loss | 4.3687 |
| 测试集 PPL | 78.94 |
| 测试集 next-word accuracy | 27.86% |
| 是否达到 PPL<80 | 是 |

训练趋势：

- Epoch 1：Train PPL 547.84，Val PPL 256.82
- Epoch 10：Train PPL 101.99，Val PPL 103.47
- Epoch 20：Train PPL 61.88，Val PPL 85.28
- Epoch 55：Train PPL 39.20，Val PPL 82.61

展示材料：

- `outputs/training_curves.png`：训练/验证 loss 和 PPL 曲线。
- `outputs/next_word_examples.md`：测试集上下文、真实目标词、Top-5 预测。
- `outputs/generated_samples.md`：给定 prompt 的 PTB 风格续写。

## 第 11 页：结果分析与总结

结果分析要点：

1. 前期 PPL 快速下降，说明模型迅速学习到高频词和句末结构。
2. Dropout=0.65 对大容量 LSTM 很重要，能缓解 PTB 小语料过拟合。
3. 梯度裁剪保证 BPTT 中 SGD 更新稳定。
4. 学习率衰减对后期突破验证 PPL 瓶颈很关键。
5. PPL 比 top-1 accuracy 更适合评价语言模型。

总结：

- 完成 PTB LSTM 语言模型训练、评估和结果输出。
- 默认 large preset 测试集 PPL=78.94，达到 PPL<80 目标。
- 可改进方向：weight tying、AWD-LSTM 正则化、Transformer 对比、top-k/top-p 解码。
