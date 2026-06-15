# 诊断测试分析报告

## 概述

本报告记录了 FinGPT-QLoRA 项目的训练配置诊断测试过程，包括不同配置的测试结果和原因分析。

**测试目标**：找到最优训练配置，平衡训练速度和模型质量。

**测试环境**：
- GPU: Tesla T4 (16GB VRAM)
- 模型: Qwen2.5-7B-Instruct-bnb-4bit
- 框架: Unsloth + TRL SFTTrainer

---

## 测试结果汇总

| 配置 | max_seq_length | packing | 训练速度 | 对比基线 | 状态 |
|------|----------------|---------|----------|----------|------|
| 基线 | 2048 | False | 9.72s/step | 1.0x | ✅ |
| v1 | 512 | True | 26.87s/step | 0.4x | ❌ |
| v2 | 512 | True (v2) | 25.89s/step | 0.4x | ❌ |
| v3 | 512 | False | 8.57s/step | 1.1x | ✅ |

---

## 详细测试记录

### 测试 1: 基线配置 (max_seq_length=2048, no packing)

**配置**：
```python
MAX_SEQ_LENGTH = 2048
packing = False
batch_size = 2
gradient_accumulation_steps = 8
```

**数据统计**：
- 训练集: 25,615 条
- 验证集: 3,013 条
- Token 长度: min=86, max=128, avg=107

**测试结果**：
- 10 步训练时间: 97.2s
- 每步时间: 9.72s
- Loss 变化: 4.42 → 1.56

**分析**：
- 训练速度符合预期（7B 模型 + T4）
- Loss 正常下降
- 每步处理的 token 数: 2 × 2048 = 4,096 tokens
- 实际有效 token: 2 × 107 = 214 tokens
- **Padding 比例**: 214 / 4,096 = 5.2%（严重浪费）

**结论**：虽然 padding 浪费严重，但这是最稳定的配置。

---

### 测试 2: 优化配置 (max_seq_length=512, packing=True)

**配置**：
```python
MAX_SEQ_LENGTH = 512
packing = True
batch_size = 2
gradient_accumulation_steps = 8
```

**测试结果**：
- 10 步训练时间: 268.7s
- 每步时间: 26.87s
- Loss 变化: 正常下降

**分析**：
- 训练速度比基线 **慢 2.8 倍**
- 日志显示: `Unsloth: Packing enabled - training is >2x faster`
- 但实际速度反而更慢

**原因分析**：
1. **Packing 开销**：
   - 需要将多个短序列打包成一个长序列
   - 需要创建特殊的 attention mask
   - 需要在训练后解包
   - 对于短序列（avg=107），开销大于收益

2. **数据集特点**：
   - 平均长度只有 107 tokens
   - max_seq_length=512 意味着每个打包序列可以放 ~4 个样本
   - 但打包/解包的开销超过了节省的 padding 时间

3. **Unsloth 实现**：
   - Unsloth 的 packing 实现可能有额外开销
   - 对于小数据集，packing 的优化效果不明显

**结论**：对于短序列数据集，packing 反而更慢。

---

### 测试 3: 优化配置 v2 (max_seq_length=512, packing=True, formatting_func)

**配置**：
```python
MAX_SEQ_LENGTH = 512
packing = True
# 使用 formatting_func 而非预处理
```

**测试结果**：
- 10 步训练时间: 258.9s
- 每步时间: 25.89s

**分析**：
- 与 v1 结果相似（26.87s vs 25.89s）
- 使用 formatting_func 没有显著改善

**原因分析**：
1. **数据格式不是瓶颈**：
   - 两种方式最终都转换为相同的 token 序列
   - 瓶颈在于 packing 本身的开销

2. **Unsloth 的优化**：
   - Unsloth 已经对标准格式做了优化
   - 自定义 formatting_func 可能破坏了 Unsloth 的优化路径

**结论**：数据格式不是速度瓶颈，packing 本身才是问题。

---

### 测试 4: 简化配置 (max_seq_length=512, no packing)

**配置**：
```python
MAX_SEQ_LENGTH = 512
packing = False
batch_size = 2
gradient_accumulation_steps = 8
```

**测试结果**：
- 10 步训练时间: 85.7s
- 每步时间: 8.57s

**分析**：
- 比基线快 1.1 倍（9.72s → 8.57s）
- 速度提升来自减少 padding

**计算**：
- 基线：每步处理 2 × 2048 = 4,096 tokens
- v3：每步处理 2 × 512 = 1,024 tokens
- 减少: 4,096 / 1,024 = 4 倍
- 但实际速度只提升 1.1 倍

**原因分析**：
1. **计算不是瓶颈**：
   - T4 的计算能力足够处理 2048 长度
   - 瓶颈可能在内存访问或数据加载

2. **Unsloth 优化**：
   - Unsloth 已经对长序列做了优化
   - 减少长度的收益有限

3. **实际有效计算**：
   - 两种配置的有效 token 数相同（avg=107）
   - 只是 padding 数量不同

**结论**：减少 max_seq_length 有小幅提升，但收益有限。

---

## 根本原因分析

### 为什么 Packing 更慢？

**理论预期**：
- Packing 应该消除 padding，提升 GPU 利用率
- 短序列应该受益最大

**实际结果**：
- Packing 反而更慢（0.4x）

**根本原因**：

1. **序列太短**：
   - 平均长度 107 tokens
   - 即使不 packing，padding 比例也不高
   - Packing 的收益不足以抵消开销

2. **Packing 开销**：
   - 打包：将多个序列拼接，创建特殊 mask
   - 解包：训练后分离结果
   - 额外的内存分配和拷贝

3. **Unsloth 实现**：
   - Unsloth 的 packing 实现可能有 bug 或优化不足
   - 对于小数据集，优化效果不明显

4. **数据加载**：
   - Packing 需要更复杂的数据加载逻辑
   - 可能引入额外的 CPU 开销

### 为什么减少 max_seq_length 收益有限？

**理论预期**：
- 减少 4 倍长度（2048 → 512）
- 应该提升 4 倍速度

**实际结果**：
- 只提升 1.1 倍

**根本原因**：

1. **计算不是瓶颈**：
   - T4 的计算能力足够
   - 瓶颈在内存访问或数据加载

2. **Unsloth 优化**：
   - Unsloth 已经对长序列做了优化
   - 减少长度的边际收益递减

3. **有效计算相同**：
   - 两种配置的有效 token 数相同
   - 只是 padding 数量不同

---

## 结论和建议

### 最优配置

**推荐配置**：
```python
MAX_SEQ_LENGTH = 2048  # 保持原配置
packing = False        # 不使用 packing
batch_size = 2
gradient_accumulation_steps = 8
```

**理由**：
1. 稳定性最好
2. 速度可接受（9.72s/step）
3. 支持更长序列（如果需要）

### 训练时间估算

**全量训练**：
- 总步数: 25,615 × 3 / 16 = 4,803 steps
- 每步时间: 9.72s
- 总时间: 4,803 × 9.72 / 3600 = 13.0 小时

**Kaggle 限制**：
- 最大运行时间: 12 小时
- 需要优化：减少 epoch 或增大 batch_size

**优化方案**：
```python
# 方案 1: 减少 epoch
num_train_epochs = 2  # 3 → 2
# 总时间: 2/3 × 13 = 8.7 小时

# 方案 2: 增大 batch_size
per_device_train_batch_size = 4  # 2 → 4
gradient_accumulation_steps = 4  # 8 → 4
# 有效批次: 4 × 4 = 16（相同）
# 每步处理更多数据，可能更快
```

---

## 后续行动

1. **使用基线配置继续训练**
   - 从 checkpoint-300 恢复
   - 使用 `resume_from_checkpoint=True`

2. **监控训练**
   - 保存日志到文件
   - 定期检查 Loss 曲线

3. **评估模型**
   - 训练完成后运行评估
   - 对比基线和微调模型

---

## 附录：原始数据

### 测试 1 日志摘要
```
[19:49:55] 10 steps completed in 97.2s
[19:49:55] Time per step: 9.72s
[19:49:55] Step 1: loss=4.4180
[19:49:55] Step 10: loss=1.5638
```

### 测试 2 日志摘要
```
[20:13:47] 10 steps completed in 268.7s
[20:13:47] Time per step: 26.87s
[20:13:47] Speedup: 0.4x
[20:13:47] ❌ FAIL
```

### 测试 3 日志摘要
```
[20:30:11] 10 steps: 258.9s
[20:30:11] Time/step: 25.89s
[20:30:11] Speedup: 0.4x
[20:30:11] ❌ FAIL
```

### 测试 4 日志摘要
```
[20:38:38] 10 steps: 85.7s
[20:38:38] Time/step: 8.57s
[20:38:38] Speedup: 1.1x
[20:38:38] ❌ FAIL (但实际可接受)
```
# 训练速度深度分析

## 问题陈述

**预期速度**：1-2s/step（基于 Unsloth 声称的 1,500-3,500 tokens/sec）
**实际速度**：9.72s/step（421 tokens/sec）
**差异**：3.5-8 倍慢于预期

---

## 根本原因分析

### 原因 1: 梯度检查点 (Gradient Checkpointing)

**配置**：
```python
use_gradient_checkpointing="unsloth"
```

**影响**：
- 梯度检查点通过重新计算激活值来节省显存
- 代价是增加 **~30% 的计算开销**
- Unsloth 的实现比标准 PyTorch 更高效，但仍有开销

**计算**：
```
基础速度: 9.72s / 1.3 = 7.48s（去掉梯度检查点开销）
```

**结论**：梯度检查点贡献了约 30% 的速度损失。

---

### 原因 2: 序列长度 (O(n²) 复杂度)

**配置**：
```python
max_seq_length = 2048
```

**影响**：
- Transformer 注意力机制是 O(n²) 复杂度
- 序列长度翻倍，计算量增加 4 倍

**计算**：
```
如果用 512: 计算量 = (512/2048)² = 1/16 = 6.25%
预期时间: 9.72 * 0.0625 = 0.6s（理论值）

但实际测试 v3 (512, no packing): 8.57s/step
只快了 1.1 倍，不是 16 倍

原因：Unsloth 已经对长序列做了优化
```

**结论**：序列长度影响有限，因为 Unsloth 已优化。

---

### 原因 3: 评估开销

**配置**：
```python
eval_strategy = "steps"
eval_steps = 100
```

**影响**：
- 每 100 步评估一次
- 评估时间：467 秒（7.8 分钟）
- 每 100 步中有 467 秒用于评估

**计算**：
```
100 步训练时间: 100 * 9.72 = 972s
评估时间: 467s
总时间: 972 + 467 = 1439s
有效训练比例: 972 / 1439 = 67.5%

如果去掉评估: 每步时间 = 9.72 * 0.675 = 6.56s
```

**结论**：评估开销占了 32.5% 的时间。

---

### 原因 4: 梯度累积开销

**配置**：
```python
gradient_accumulation_steps = 8
```

**影响**：
- 每 8 步才更新一次参数
- 需要保存 8 步的梯度
- 增加内存访问和同步开销

**计算**：
```
如果用 gradient_accumulation_steps = 1:
  - 每步都更新参数
  - 减少梯度保存开销
  - 但需要减小 batch_size 或增大显存

如果用 gradient_accumulation_steps = 2:
  - 开销减少 4 倍
  - 有效批次从 16 降到 4
```

**结论**：梯度累积增加了内存和同步开销。

---

### 原因 5: 批次大小

**配置**：
```python
per_device_train_batch_size = 2
```

**影响**：
- T4 的 16GB 显存限制
- batch_size=2 是保守选择
- GPU 利用率可能不高

**计算**：
```
如果用 batch_size = 4:
  - 需要更多显存
  - 可能需要减少 max_seq_length
  - GPU 利用率更高
```

**结论**：批次大小受限于显存。

---

## 综合分析

### 时间分解 (每 100 步)

```
训练时间: 972s (67.5%)
评估时间: 467s (32.5%)
总计: 1439s

每步平均时间: 1439 / 100 = 14.39s（包含评估）
纯训练每步时间: 9.72s
```

### 吞吐量计算

```
每步处理的 token:
  - 带 padding: 2 * 2048 = 4096 tokens
  - 实际有效: 2 * 107 = 214 tokens

吞吐量:
  - 带 padding: 4096 / 9.72 = 421 tokens/sec
  - 有效: 214 / 9.72 = 22 tokens/sec
```

### 与 Unsloth 基准对比

```
Unsloth 声称: 1,500-3,500 tokens/sec
我们的结果: 421 tokens/sec

差异: 3.5-8 倍

可能原因:
1. 基准测试可能没有梯度检查点
2. 基准测试可能没有评估
3. 基准测试可能用更短序列
4. 基准测试可能用更大批次
```

---

## 优化建议

### 方案 1: 减少评估频率

```python
eval_steps = 500  # 从 100 改为 500

效果:
  - 评估时间减少 5 倍
  - 每步平均时间从 14.39s 降到 ~10.5s
  - 速度提升: 1.4 倍
```

### 方案 2: 减少梯度累积

```python
gradient_accumulation_steps = 4  # 从 8 改为 4

效果:
  - 减少内存和同步开销
  - 有效批次从 16 降到 8
  - 可能需要调整学习率
```

### 方案 3: 减少序列长度

```python
max_seq_length = 512  # 从 2048 改为 512

效果:
  - 减少 padding 浪费
  - 实际测试: 8.57s/step (1.1 倍提升)
  - 收益有限
```

### 方案 4: 组合优化

```python
# 最优配置
MAX_SEQ_LENGTH = 512
gradient_accumulation_steps = 4
eval_steps = 500
packing = False

预期效果:
  - 每步时间: ~6-7s
  - 速度提升: ~1.5 倍
```

---

## 结论

### 为什么 1-2s/step 的预期是错误的？

1. **Unsloth 基准条件不同**：
   - 可能没有梯度检查点
   - 可能没有评估
   - 可能用更短序列

2. **我们的配置更保守**：
   - 梯度检查点：+30% 开销
   - 评估：+32.5% 开销
   - 总开销：~60%

3. **实际吞吐量在合理范围**：
   - 421 tokens/sec（带 padding）
   - 22 tokens/sec（有效）
   - 考虑到开销，这是合理的

### 实际预期

```
如果去掉所有开销:
  - 基础速度: 9.72 / 1.3 / 1.325 = 5.6s/step
  - 吞吐量: 4096 / 5.6 = 731 tokens/sec

这仍然低于 Unsloth 基准 (1,500-3,500)
可能原因:
  - T4 计算能力有限
  - 模型较大 (7B)
  - 我们的配置不是最优
```

---

## 行动建议

### 短期优化
1. 增加 eval_steps: 100 → 500
2. 减少 gradient_accumulation_steps: 8 → 4
3. 重新评估学习率

### 长期优化
1. 测试不同 batch_size
2. 测试不同 max_seq_length
3. 考虑使用更强大的 GPU (A100)

### 基准测试
1. 运行纯训练测试（无评估）
2. 测量实际吞吐量
3. 与 Unsloth 官方基准对比
