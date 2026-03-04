# RAG项目重构计划

## 目标
1. 数据与代码解耦
2. 构建完整RAG pipeline（召回+精排+生成）
3. 支持embedding模型微调
4. 提供前端展示界面

## 当前问题
- 代码和数据混在一起（OldTrain、NewTrain、ORIE-5981-RAG）
- 大量重复代码分散在多个notebook中
- 配置硬编码，难以管理不同实验
- 缺少reranker精排阶段
- 没有统一的前端界面

## 新项目结构

```
Capstone-Project/
├── data/                    # 所有数据集
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后的数据
│   └── fine_tune/           # 微调数据
├── src/                     # 核心代码模块
│   ├── config/              # 配置管理
│   ├── data/                # 数据加载和预处理
│   ├── models/              # Embedding模型
│   ├── retrieval/           # 召回+精排(reranker)
│   ├── generation/          # RAG生成
│   ├── evaluation/          # 评估指标
│   └── app/                 # Gradio前端
├── notebooks/               # Colab notebooks
│   ├── launcher.ipynb       # 主启动器
│   └── experiments/         # 各种实验notebook
├── experiments/             # 实验配置文件(yaml)
├── models/                  # 训练好的模型
└── outputs/                 # 运行结果

```

## 核心模块

### 1. 配置系统 (src/config/)
- 统一管理路径、模型、超参数
- YAML配置文件，支持多实验

### 2. 数据模块 (src/data/)
- 统一的数据加载接口
- 支持hospital和corruption两个数据集

### 3. 检索模块 (src/retrieval/)
- **召回**: FAISS向量检索 (top-50)
- **精排**: Cross-Encoder reranker (top-10)
- 支持有/无reranker两种模式

### 4. 生成模块 (src/generation/)
- Azure OpenAI调用
- Prompt模板管理

### 5. 前端 (src/app/)
- Gradio界面
- 支持Colab内运行 + 本地运行

## 实现步骤

### Phase 1: 基础架构
- [ ] 创建目录结构
- [ ] 迁移和整理数据文件
- [ ] 编写配置系统

### Phase 2: 核心模块
- [ ] 数据加载模块
- [ ] FAISS索引构建
- [ ] 基础检索器
- [ ] RAG生成器

### Phase 3: Reranker集成
- [ ] 实现Cross-Encoder reranker
- [ ] 混合检索pipeline (召回+精排)
- [ ] 评估reranker效果

### Phase 4: 微调系统
- [ ] 重构fine-tune代码
- [ ] 支持不同微调策略
- [ ] 训练脚本

### Phase 5: 前端和Demo
- [ ] Gradio界面开发
- [ ] Colab launcher notebook
- [ ] 使用文档

## Colab工作流
- 本地开发src/模块代码
- 使用本地连接的Colab运行GPU任务
- Launcher notebook作为统一入口
- 支持后续迁移到共享Colab

## 技术栈
- **Embedding**: Qwen3-Embedding-0.6B + LoRA微调
- **向量库**: FAISS
- **Reranker**: sentence-transformers CrossEncoder
- **生成**: Azure OpenAI GPT-4
- **前端**: Gradio
- **配置**: YAML + dataclass
