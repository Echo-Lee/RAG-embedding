# 📊 模型对比功能使用指南

## 概述

新版 Gradio 应用支持**双栏对比**功能，可以同时比较：
- 微调前（Base Model）vs 微调后（Fine-tuned Model）
- 不同微调版本之间的效果对比

---

## 功能特点

### 单模型模式
- 选择一个模型查询
- 显示该模型的检索结果

### 对比模式 ⭐
- **双栏布局**：左右并排显示两个模型结果
- **同时检索**：一次查询，两个模型同时工作
- **对比指标**：
  - Top Score（最高分数）
  - Average Score（平均分数）
  - Top K 结果列表

---

## 配置步骤

### Step 1: 训练 Fine-tuned 模型

在 Colab 中训练你的模型（使用 LoRA 或 Full Fine-tuning）：

```python
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, TaskType

# 加载 base model
base_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

# 训练...（你的训练代码）

# 保存 fine-tuned 模型
model.save_pretrained("./finetuned-hospital-v1")

# 上传到 HF Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./finetuned-hospital-v1",
    repo_id="你的用户名/rag-finetuned-hospital",
    repo_type="model"
)
```

### Step 2: 配置模型列表

编辑 `gradio_app_compare.py` 第 16-29 行：

```python
MODEL_CONFIGS = {
    'base': {
        'name': 'Base Model',
        'model_id': 'Qwen/Qwen3-Embedding-0.6B',
        'description': 'Pre-trained Qwen3 embedding model'
    },
    'finetuned-hospital': {  # ← 添加你的 fine-tuned 模型
        'name': 'Fine-tuned (Hospital)',
        'model_id': '你的用户名/rag-finetuned-hospital',  # ⚠️ 改成你的模型
        'description': 'Fine-tuned on hospital emails'
    },
    # 可以添加更多模型...
    # 'finetuned-v2': {
    #     'name': 'Fine-tuned v2',
    #     'model_id': '你的用户名/rag-finetuned-v2',
    #     'description': 'Second fine-tuned version'
    # },
}
```

### Step 3: 部署到 HF Space

1. 复制 `gradio_app_compare.py` 内容
2. 在 HF Space 创建/更新 `app.py`
3. 修改：
   - 第 11 行：`YOUR_HF_USERNAME`
   - 第 16-29 行：`MODEL_CONFIGS`（添加你的模型）
4. 保存，等待构建

---

## 使用方法

### 对比模式使用

1. **选择模式**: 点击 "Compare Models"
2. **选择数据集**: hospital 或 corruption
3. **选择两个模型**:
   - Model 1 (左侧): base
   - Model 2 (右侧): finetuned-hospital
4. **输入问题**: "What are the main issues?"
5. **点击 Compare**: ⚖️

### 结果对比

**左侧（Base Model）**:
```
Model: Base Model
Top Score: 0.7523
Avg Score: 0.6891

1. Score: 0.7523
   Index: 1234
---
2. Score: 0.7102
   Index: 5678
---
...
```

**右侧（Fine-tuned Model）**:
```
Model: Fine-tuned (Hospital)
Top Score: 0.8912  ← 更高！
Avg Score: 0.8234  ← 更高！

1. Score: 0.8912
   Index: 1234
---
2. Score: 0.8456
   Index: 9012
---
...
```

**关键指标**：
- ✅ Fine-tuned 模型的分数更高
- ✅ 检索更精准
- ✅ 相关文档排名更靠前

---

## 对比示例

### 示例 1: 专业术语查询

**Query**: "What is the status of the litigation?"

| 指标 | Base Model | Fine-tuned | 改进 |
|------|-----------|-----------|-----|
| Top Score | 0.723 | 0.891 | +23% |
| Avg Score | 0.654 | 0.812 | +24% |
| 相关文档排名 | #3 | #1 | ⬆️ |

### 示例 2: 复杂问题

**Query**: "Summarize discussions about patient safety protocols"

| 指标 | Base Model | Fine-tuned | 改进 |
|------|-----------|-----------|-----|
| Top Score | 0.698 | 0.856 | +23% |
| Avg Score | 0.621 | 0.789 | +27% |

---

## 高级用法

### 对比多个 Fine-tuned 版本

如果你训练了多个版本：

```python
MODEL_CONFIGS = {
    'base': {...},
    'finetuned-v1': {
        'name': 'Fine-tuned v1 (LoRA)',
        'model_id': '你的用户名/rag-finetuned-v1',
    },
    'finetuned-v2': {
        'name': 'Fine-tuned v2 (Full)',
        'model_id': '你的用户名/rag-finetuned-v2',
    },
}
```

然后对比：
- v1 vs v2（比较不同训练方法）
- base vs v2（比较最终效果）

### 评估指标

**好的 fine-tuned 模型应该**：
- ✅ Top Score 提升 15-30%
- ✅ Average Score 提升 20-35%
- ✅ 相关文档排名更靠前
- ✅ 对专业术语理解更好

---

## 常见问题

### Q1: 只有 base model，没有 fine-tuned？

**A**: 应用会自动检测：
- 1 个模型 → 默认单模型模式
- 2+ 个模型 → 默认对比模式

你可以先用单模型模式，训练完再添加 fine-tuned 模型。

### Q2: 如何添加第三个模型？

**A**: 在 `MODEL_CONFIGS` 中添加即可：

```python
'finetuned-v3': {
    'name': 'Fine-tuned v3',
    'model_id': '你的用户名/rag-finetuned-v3',
    'description': 'Third version'
}
```

然后在对比时从下拉菜单选择。

### Q3: 对比很慢怎么办？

**A**: 两个模型同时运行确实会慢一些：
- CPU: ~5-8 秒
- GPU T4: ~2-3 秒

建议升级到 GPU Space（免费）。

### Q4: 如何保存对比结果？

**A**: 目前支持：
- 复制文本结果
- 截图保存
- 未来可以添加导出 CSV 功能

---

## 部署检查清单

部署对比版本前，确保：

- [ ] 已训练 fine-tuned 模型
- [ ] 模型已上传到 HF Hub
- [ ] 修改了 `YOUR_HF_USERNAME`
- [ ] 配置了 `MODEL_CONFIGS`（添加你的模型）
- [ ] 索引已上传到 HF Hub
- [ ] HF Space 构建成功

---

## 文件对比

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `gradio_app.py` | 单模型 | 只有 base model |
| `gradio_app_compare.py` | 对比模式 | 有 fine-tuned model |

**推荐**: 如果计划训练模型，直接用 `gradio_app_compare.py`，向前兼容。

---

## 示例工作流

### Week 1: Base Model
1. 部署 `gradio_app_compare.py`（只配置 base model）
2. 测试基准效果
3. 收集问题案例

### Week 2: Fine-tuning
1. 在 Colab 训练 fine-tuned 模型
2. 上传到 HF Hub
3. 更新 Space 的 `MODEL_CONFIGS`

### Week 3: 对比评估
1. 使用对比模式测试
2. 记录指标改进
3. 展示给用户/导师

---

## 技术细节

### 对比算法

```python
# 同时调用两个模型
docs1 = retrieve_with_model(query, dataset, model1, top_k)
docs2 = retrieve_with_model(query, dataset, model2, top_k)

# 计算统计指标
top_score_1 = docs1[0]['score']
avg_score_1 = mean([d['score'] for d in docs1])

top_score_2 = docs2[0]['score']
avg_score_2 = mean([d['score'] for d in docs2])

# 双栏显示
display_side_by_side(docs1, docs2)
```

### 性能优化

如果对比慢，可以：
1. 减少 `top_k`（默认 10）
2. 禁用 reranker（对比时）
3. 使用 GPU Space

---

## 总结

**双栏对比功能**让你可以：
- ✅ 直观比较微调效果
- ✅ 量化改进指标
- ✅ 展示给老师/用户
- ✅ 调试模型表现

**开始使用**：
1. 先用 base model 部署
2. 训练 fine-tuned 模型
3. 添加到配置
4. 开始对比！

---

需要帮助？查看 [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) 或提问！
