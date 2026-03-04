# 本地编辑 + Colab执行 工作流

## 🎯 核心思路

你可以在**本地舒服地编辑代码**，同时使用**Colab的GPU和依赖包**执行！

| 组件 | 位置 | 说明 |
|------|------|------|
| **代码编辑** | 本地IDE | VSCode/PyCharm等 |
| **代码存储** | GitHub | 版本控制 |
| **执行环境** | Colab | GPU + 预装包 |
| **数据** | Google Drive | 持久化 |
| **Outputs** | Google Drive | 自动保存 |

---

## 📦 一次性准备

### 1. 上传数据到Google Drive

```
Google Drive/
└── Capstone-Data/
    ├── hospital/
    │   └── threads_with_summary.json        (28MB)
    └── corruption/
        └── emails_group_by_thread.json      (3.3MB)
```

### 2. 配置Colab Secrets（一次性）

在Colab中设置：
1. 点击左侧栏 🔑 **Secrets**
2. 添加两个secrets:
   - `AZURE_API_KEY`: `your-api-key`
   - `AZURE_ENDPOINT`: `your-endpoint`

### 3. （可选）连接Colab到本地

如果用VSCode：
- 安装 "Colab" 扩展
- 或用 Jupyter extension + Colab runtime URL

---

## 🚀 日常工作流

### Step 1: 首次运行 - 执行Setup

在本地打开 **`local_colab_setup.ipynb`**，连接到Colab runtime，运行所有cells。

这个notebook会：
1. ✅ 从GitHub拉取代码到 `/content/RAG-embedding/`
2. ✅ 安装依赖（sentence-transformers, faiss, gradio等）
3. ✅ 挂载Google Drive
4. ✅ 链接数据文件
5. ✅ 链接outputs到Drive（自动保存）
6. ✅ 配置API keys
7. ✅ 添加到Python path

**只需运行一次！** 之后就可以直接import项目模块。

---

### Step 2: 运行Pipeline

打开 **`example_pipeline.ipynb`** 或创建你自己的notebook：

```python
# 选择模式和数据集
MODE = "full"          # "full" = 构建索引, "quick" = 加载已有索引
DATASET = "hospital"   # "hospital" 或 "corruption"

# 加载配置
from config.config import load_config
config = load_config(f'experiments/{DATASET}_base.yaml')

# 构建或加载索引
if MODE == "full":
    from data.loader import DataLoader
    from retrieval.indexer import FAISSIndexer

    loader = DataLoader(config)
    documents = loader.load()

    indexer = FAISSIndexer(config)
    index = indexer.build_index(documents)
    indexer.save_index(f"outputs/indexes/{DATASET}")  # 自动保存到Drive

# 初始化检索器和生成器
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.rag_generator import AzureRAGGenerator

retriever = HybridRetriever(config, index_path=f"outputs/indexes/{DATASET}")
reranker = CrossEncoderReranker(config) if config.use_reranker else None
generator = AzureRAGGenerator(config)

# 测试查询
query = "What are the main issues?"
docs = retriever.retrieve(query, top_k=10, use_rerank=True)
answer = generator.generate(query, docs)
print(answer)

# 或启动Gradio demo
from app.gradio_app import create_demo
demo = create_demo(retriever, reranker, generator, config)
demo.launch(share=True)
```

---

### Step 3: 修改代码

#### 在本地修改
```bash
# 本地编辑 src/retrieval/retriever.py 或其他文件

# 提交到GitHub
git add .
git commit -m "Improved retrieval logic"
git push
```

#### 在Colab中更新
```python
# 在notebook的一个cell中运行
!cd /content/RAG-embedding && git pull

# 重启kernel（或重新import模块）
```

---

## 🎯 完整示例场景

### 场景1: 首次运行（构建索引）

```python
# 1. 运行 local_colab_setup.ipynb（一次性）

# 2. 在example_pipeline.ipynb中：
MODE = "full"
DATASET = "hospital"
# 运行所有cells
# 等待15分钟构建索引
# 索引自动保存到Drive
```

### 场景2: 后续运行（快速加载）

```python
# Setup已完成，直接运行：
MODE = "quick"  # ⚡ 1秒加载
DATASET = "hospital"
# 立即启动demo
```

### 场景3: 修改代码后测试

```bash
# 本地
git push

# Colab notebook cell
!cd /content/RAG-embedding && git pull
# 重启kernel，重新运行
```

### 场景4: Session断开后恢复

```python
# 重新连接Colab
# 运行 local_colab_setup.ipynb（重新setup）
# 然后用MODE="quick"快速加载索引
# 无需重建索引！
```

---

## 📁 目录结构

### 本地项目

```
Capstone Project/
├── local_colab_setup.ipynb       # ⭐ Setup notebook（首次运行）
├── example_pipeline.ipynb        # ⭐ 示例pipeline
├── your_notebook.ipynb           # 你自己的notebook
├── src/                          # 代码（会被push到GitHub）
│   ├── config/
│   ├── data/
│   ├── retrieval/
│   ├── generation/
│   └── app/
├── experiments/                  # 配置
│   ├── hospital_base_template.yaml
│   └── corruption_base_template.yaml
└── notebooks/                    # 其他notebooks
```

### Colab Runtime

```
/content/
├── RAG-embedding/                # 从GitHub clone的代码
│   ├── src/
│   ├── experiments/
│   ├── data/                     # → 链接到Drive
│   └── outputs/                  # → 链接到Drive
└── drive/
    └── MyDrive/
        ├── Capstone-Data/        # 你的数据
        └── Capstone-Outputs/     # 自动保存的索引
```

---

## ✨ 关键优势

### 相比纯Colab Web

| 项目 | 纯Web Colab | 本地编辑 + Colab |
|------|------------|-----------------|
| **编辑体验** | Web编辑器（受限） | 本地IDE（舒适） ✅ |
| **代码补全** | 基础 | 完整IDE支持 ✅ |
| **多文件编辑** | 切换麻烦 | 轻松切换 ✅ |
| **版本控制** | 手动 | Git集成 ✅ |
| **执行环境** | Colab GPU | Colab GPU |
| **依赖管理** | 每次安装 | 一次setup |

---

## 📝 常见问题

### Q: 为什么不直接在本地运行？
A: 因为需要GPU和大量依赖包。本地可能没有GPU，或者配置环境麻烦。

### Q: 修改代码后必须git push吗？
A: 是的。因为代码在GitHub，Colab通过git pull获取最新代码。这样也方便版本控制。

### Q: 每次连接Colab都要运行setup吗？
A: 是的，因为Colab runtime是临时的。但setup很快（1-2分钟），而且**索引在Drive里不会丢失**。

### Q: 能不能不用GitHub，直接上传代码到Colab？
A: 可以，但不推荐。用GitHub的好处：
  - 版本控制
  - 多人协作方便
  - 只需`git pull`更新，不用重复上传

### Q: VSCode怎么连接Colab？
A:
  1. 安装VSCode的Colab扩展
  2. 或用Jupyter扩展连接Colab runtime URL
  3. 参考: https://code.visualstudio.com/docs/datascience/jupyter-notebooks

---

## 🎉 总结

**你的工作流**：

```
本地IDE（编辑） → Git（同步） → Colab（执行） → Drive（保存）
     ↑                                              ↓
     └──────────── 修改代码，git pull更新 ──────────┘
```

**核心文件**：
- `local_colab_setup.ipynb` - 一次性setup
- `example_pipeline.ipynb` - 运行pipeline示例
- 你自己的notebooks - 自由发挥

**记住**：
- ✅ 代码在本地编辑，push到GitHub
- ✅ 在Colab中`git pull`更新
- ✅ 数据和outputs在Drive，永久保存
- ✅ Setup只需运行一次（每个session）

---

**享受本地编辑的舒适 + Colab GPU的强大！** 🚀
