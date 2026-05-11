# 实验四 基于 Transformer 的中→英神经机器翻译 · PPT 提纲

> 共 10 页，建议每页 1-2 分钟。封面 + 目录 + 8 内容页。

---

## P1 · 封面

- 标题：**基于 Transformer 的中英神经机器翻译**
- 副标题：从零实现 Encoder–Decoder + BPE 子词分词
- 姓名 / 学号 / 课程 / 日期

## P2 · 目录

1. 任务概述
2. 数据集介绍
3. Transformer 模型结构
4. 损失函数与优化器
5. 训练过程与配置
6. 实验结果
7. 结果分析与讨论
8. 总结与改进方向

## P3 · 任务概述

- **任务定义**：把中文句子翻译为英文（Chinese→English NMT）
- **应用场景**：跨语言信息检索、跨境电商、新闻翻译、多语言客服
- **核心挑战**：变长输入输出、长距离语义依赖、低资源词汇 OOV
- **本实验目标**：Test BLEU-4 ≥ 14

## P4 · 数据集介绍

- **来源**：NiuTrans 中英平行语料（东北大学 NLP Lab）
- **规模**：训练 99,000 句对 / 验证 1,000 / 测试 1,000
- **特点**：中文已 ICTCLAS 分词、英文已 tokenize 并 lower-case
- **预处理修复**：原始 `dev.zh / dev.en` 行错位、`test.en` 内容被误置成中文，本实验用 `_parse_paired_blocks` 重新解析三行块（zh ⏎ blank ⏎ en）
- **分词**：双 SentencePiece **BPE**（zh & en 各 16,000 词），OOV ≈ 0%
- **样例展示**：1 条原始中文 ⇒ 1 条英文参考

## P5 · 模型结构设计（重点页）

> 配 Transformer 架构图（左 Encoder × 6，右 Decoder × 6，中间 Cross-Attention 连接）。

- **整体**：Encoder–Decoder，6+6 层，d_model=256，heads=8，d_ff=1024，dropout=0.1。
- **Pre-LN 残差**：`x = x + Dropout(SubLayer(LayerNorm(x)))` —— 训练更稳。
- **核心模块**：Multi-Head Self-Attention / Cross-Attention / FeedForward / 正余弦位置编码。
- **权重共享**：目标词嵌入与输出投影共享权重（weight tying）。
- **Mask 语义关键点**：`F.scaled_dot_product_attention` 用 **True=attend** 约定（与 `nn.MultiheadAttention` 相反）；调试中曾因此 BLEU=0 的踩坑。
- **总参数量**：19.25 M

## P6 · 损失函数与优化器

- **损失**：CrossEntropy + label smoothing（ε=0.1），忽略 PAD
- **优化器**：AdamW（β=(0.9, 0.98)，ε=1e-9，wd=1e-5）
- **学习率**：warmup-rsqrt（peak=7e-4，warmup=2000 步）
- **梯度裁剪**：`clip_grad_norm_=1.0`
- **混合精度**：bfloat16 AMP

## P7 · 训练过程与配置

| 超参 | 值 |
|------|----|
| Batch size | 128 |
| Epoch | 30 |
| 输入长度上限 | 128 BPE tokens |
| Beam size | 5（length penalty 0.6） |
| 训练硬件 | RTX 5070 (12 GB) |
| 总训练时间 | ~30 min |

> 建议附一张实时训练截图：GPU 利用率 / 单 epoch 耗时。

## P8 · 实验结果

> 配 `outputs/training_curves.png`（loss/ppl/bleu/lr 四象限）。

| 指标 | 数值 |
|------|------|
| Best Val BLEU-4 | **26.69**（epoch 28） |
| Test BLEU-4（greedy） | **26.86** |
| Test BLEU-4（beam=5） | **27.79** |
| Test loss / ppl | 3.117 / 22.58 |
| 实验目标 | ≥ 14 ✅ |

- BLEU 增长拐点：**epoch 4** 即超过目标 14 → epoch 25 收敛在 26 附近。

## P9 · 翻译样例与结果分析

**翻译样例**（节选自 `outputs/test_samples.json`）：

- ✅ 「second , comprehensive management .」（与参考完全一致）
- ✅ 「tunisia highly praised china 's foreign and domestic policies …」（语序略改，语义一致）
- ⚠️ 「article 20 the people 's bank of china set up a renminbi issuance facility …」（出现 "branch branch branches" 重复）
- ❌ 「shi guangsheng expressed his thanks for this .」（漏译"his support"）

**关键发现**：

1. **mask 方向修复**：把 `F.scaled_dot_product_attention` 的 mask 语义改正后，BLEU 从 0 立刻跃升到正常水平。
2. **BPE 是必要条件**：词级别会有 38–51% OOV，BPE 后近乎 0%。
3. **Beam=5 比 greedy 多 0.93 BLEU**。
4. **后期收益递减**：epoch 20 已达 25.3，再训 10 epoch 仅多 ~1 BLEU。

## P10 · 总结与改进方向

**完成情况**

- ✅ 从零实现 Transformer NMT 全套组件
- ✅ 修复了原始数据流水线的两个 bug（行错位 + 测试集内容错误）
- ✅ Test BLEU 27.79，**接近 2 倍**于实验目标

**改进方向**

1. 模型放大到 Transformer-base（d_model=512，~65 M params） → 期望 ~30 BLEU
2. 数据增广：回译（Back-Translation）+ 域内单语数据
3. 解码增强：no-repeat-ngram、coverage penalty、ensemble
4. 用 mBART / NLLB 等预训练模型微调
5. 多参考 BLEU 评估（当前测试集仅 1 参考偏严苛）
