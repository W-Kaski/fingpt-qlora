# QLoRA 训练规范文档

## 一、训练前准备

### 1.1 环境检查
- [ ] GPU 型号和 VRAM 确认（T4: 16GB）
- [ ] PyTorch 和 CUDA 版本兼容性
- [ ] Unsloth 版本（推荐最新版）

### 1.2 数据准备
- [ ] 数据格式：对话格式（conversational）或文本格式
- [ ] 数据质量检查：去重、过滤短文本、清洗
- [ ] 数据集划分：train/val/test（85/10/5）

### 1.3 模型选择
- [ ] 基础模型：Qwen2.5-7B-Instruct-bnb-4bit
- [ ] 量化方式：4-bit NF4（QLoRA 标准）

---

## 二、超参数配置

### 2.1 LoRA 配置
```python
LoRAConfig(
    r=16,                          # 秩：8-64，16 是常用值
    lora_alpha=32,                 # 通常 alpha = 2 * r
    lora_dropout=0,                # dropout: 0-0.1
    target_modules=[               # 目标模块：所有线性层
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
)
```

### 2.2 训练配置
```python
SFTConfig(
    # 批次配置
    per_device_train_batch_size=2,  # 根据 VRAM 调整
    gradient_accumulation_steps=8,  # 有效批次 = 2 * 8 = 16
    
    # 学习率
    learning_rate=2e-4,             # LoRA 推荐 1e-4 到 2e-4
    lr_scheduler_type="cosine",     # 余弦退火
    warmup_ratio=0.05,              # 5% 预热
    weight_decay=0.01,
    
    # 训练轮数
    num_train_epochs=3,             # 3-5 轮
    
    # 序列长度
    max_seq_length=2048,            # 根据数据分布设置
    
    # 优化器
    optim="adamw_8bit",             # 8-bit AdamW 节省显存
    
    # 精度
    fp16=True,                      # 或 bf16=True（推荐 bf16）
    
    # 日志和保存
    logging_steps=5,
    save_steps=100,
    save_total_limit=3,             # 最多保存 3 个 checkpoint
    eval_strategy="steps",
    eval_steps=100,
    
    # 其他
    seed=42,
    report_to="none",               # 或 "wandb"
    dataset_text_field="text",      # 数据集字段名
    packing=False,                  # 是否启用 packing
)
```

---

## 三、训练流程

### 3.1 诊断测试（必须）
在全量训练前，先运行诊断测试验证配置：

```python
# 诊断测试：10 步
training_args.max_steps = 10
training_args.save_strategy = "no"
training_args.eval_strategy = "no"

# 检查指标
# - 训练速度：< 10s/step（T4 + 7B）
# - Loss 下降：应该逐步下降
# - VRAM 使用：不超过 15GB
```

### 3.2 全量训练
```python
# 从诊断测试确认配置后，运行全量训练
training_args.max_steps = -1  # 不限制步数
training_args.num_train_epochs = 3

# 继续训练（如果有 checkpoint）
trainer.train(resume_from_checkpoint=True)
```

### 3.3 监控指标
- **训练 Loss**：应该逐步下降
- **评估 Loss**：不应该持续上升（过拟合信号）
- **学习率**：按余弦曲线变化
- **梯度范数**：不应该出现爆炸

---

## 四、Kaggle 最佳实践

### 4.1 Notebook 管理
- **不要创建多个 Notebook**：使用 Kaggle 的版本管理
- **版本命名**：`v1_baseline`, `v1_fix_data`, `v2_high_lr` 等
- **复用 checkpoint**：使用 `resume_from_checkpoint=True`

### 4.2 日志记录
```python
# 保存日志到文件（Kaggle CLI 日志有延迟）
import sys

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open("results/training_log.txt", "w")
sys.stdout = Tee(sys.__stdout__, log_file)
```

### 4.3 时间估算
- **诊断测试**：~5 分钟
- **全量训练**：4,800 steps × 10s/step = ~13 小时
- **Kaggle 限制**：12 小时（需要优化配置）

---

## 五、诊断分析方法

### 5.1 训练速度分析
```python
# 检查点
1. Token 长度分布（min, max, avg）
2. Padding 比例（avg_length / max_seq_length）
3. 每步时间 vs 预期时间

# 速度慢的原因
- 序列太长：max_seq_length >> avg_token_length
- Packing 开销：packing=True 有时反而更慢
- 显存不足：频繁换页
```

### 5.2 Loss 分析
```python
# 正常情况
- 训练 Loss：逐步下降
- 评估 Loss：先降后平（或略升）

# 异常情况
- Loss 不降：学习率太低、数据问题
- Loss 爆炸：学习率太高、梯度爆炸
- 评估 Loss 上升：过拟合
```

---

## 六、常见问题

### 6.1 Packing 不生效
```
Unsloth: Sample packing skipped (custom data collator detected)
```
**原因**：自定义数据处理器冲突
**解决**：使用 `dataset_text_field` 而非自定义 collator

### 6.2 训练速度慢
**检查清单**：
- [ ] max_seq_length 是否合理（不应远大于平均长度）
- [ ] batch_size 是否最优（T4 推荐 2-4）
- [ ] 是否启用了 gradient_checkpointing
- [ ] 是否使用了 Unsloth 优化

### 6.3 显存不足
**解决方案**：
- 减小 batch_size
- 减小 max_seq_length
- 启用 gradient_checkpointing
- 使用 8-bit 优化器

---

## 七、检查清单

### 训练前
- [ ] 数据格式正确（对话格式或文本格式）
- [ ] 数据已清洗和划分
- [ ] 诊断测试通过（速度、Loss）
- [ ] 确认 checkpoint 路径

### 训练中
- [ ] 监控 Loss 曲线
- [ ] 检查 VRAM 使用
- [ ] 定期保存 checkpoint

### 训练后
- [ ] 评估模型效果
- [ ] 保存最终模型
- [ ] 记录训练日志

---

## 八、参考资源

- [HuggingFace SFT Trainer 文档](https://huggingface.co/docs/trl/en/sft_trainer)
- [Unsloth 官方文档](https://unsloth.ai/docs)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
