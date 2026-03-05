# 🚀 Gradio Space 部署指南（2步搞定）

## 总览

**目标**: 在 HuggingFace Space 部署 RAG 系统
**时间**: 不到 10 分钟
**成本**: 免费

---

## 前置准备

### 1. 注册 HuggingFace 账号

如果还没有账号：
1. 访问 https://huggingface.co/join
2. 填写信息注册
3. 验证邮箱

### 2. 获取 Access Token

1. 登录后访问 https://huggingface.co/settings/tokens
2. 点击 **"New token"**
3. 填写：
   - Name: `rag-upload`（或任意名字）
   - Role: 选择 **Write**
4. 点击 **"Generate a token"**
5. **⚠️ 复制并保存这个 token**（只显示一次）

---

## Step 1: 上传索引到 HuggingFace Hub

### 1.1 打开 Colab

在 Colab 中打开一个新 notebook。

### 1.2 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1.3 运行上传脚本

复制 `upload_to_hf.py` 的内容到 Colab，**修改第 13 行**：

```python
YOUR_HF_USERNAME = "你的HF用户名"  # ⚠️ 改成你的用户名
```

然后运行脚本：
- 会提示输入 HF Token
- 粘贴刚才保存的 token
- 自动上传索引

**预期输出**：
```
🚀 HuggingFace Hub 索引上传工具
📝 Step 1: Login to HuggingFace Hub
✅ 登录成功！
📦 Step 2: Creating repository...
✅ Repository 已创建
📤 Step 3.1: Uploading hospital index
✅ hospital 上传成功！
📤 Step 3.2: Uploading corruption index
✅ corruption 上传成功！
🎉 上传完成！
```

### 1.4 验证上传

访问：`https://huggingface.co/datasets/你的用户名/rag-indexes`

应该看到：
```
rag-indexes/
├── hospital/
│   ├── faiss_index.bin
│   └── metadata.json
└── corruption/
    ├── faiss_index.bin
    └── metadata.json
```

---

## Step 2: 创建 Gradio Space

### 2.1 创建 Space

1. 访问 https://huggingface.co/spaces
2. 点击右上角 **"Create new Space"**
3. 填写信息：
   - **Owner**: 你的用户名
   - **Space name**: `rag-search`（或任意名字）
   - **License**: MIT
   - **Space SDK**: 选择 **Gradio**
   - **Space hardware**: **CPU basic - Free**
   - **Visibility**: Public
4. 点击 **"Create Space"**

### 2.2 添加代码文件

#### 方法 1：Web 界面（推荐）

1. 在 Space 页面，点击 **"Files"** 标签
2. 点击 **"Add file"** → **"Create a new file"**
3. 文件名填：`app.py`
4. 复制 `gradio_app.py` 的**完整内容**
5. **⚠️ 重要**：修改第 11 行：
   ```python
   YOUR_HF_USERNAME = "你的HF用户名"  # 改成你的用户名
   ```
6. 点击 **"Commit new file to main"**

#### 方法 2：Git（进阶）

```bash
# 克隆 Space repository
git clone https://huggingface.co/spaces/你的用户名/rag-search
cd rag-search

# 复制文件
cp ../gradio_app.py app.py

# 修改 YOUR_HF_USERNAME
# 编辑 app.py 第 11 行

# 提交
git add app.py
git commit -m "Add RAG search application"
git push
```

### 2.3 等待构建

保存后，Space 会自动开始构建：
- 顶部状态：🟡 **Building...** （2-5分钟）
- 构建完成：🟢 **Running**

**查看日志**：
- 点击 **"Logs"** 标签查看构建进度
- 应该看到：
  ```
  🔄 Loading resources from HuggingFace Hub...
  📥 Loading hospital index...
    ✅ hospital: 9300 documents
  📥 Loading corruption index...
    ✅ corruption: 800 documents
  🤖 Loading embedding model...
  ✅ Loaded 1 models: ['base']
  🎉 All resources loaded! Backend ready.
  ```

---

## Step 3: 测试

### 3.1 打开应用

构建完成后，顶部会显示：
```
🟢 Running
```

访问你的 Space URL：
```
https://huggingface.co/spaces/你的用户名/rag-search
```

### 3.2 测试查询

1. **选择数据集**: hospital
2. **选择模型**: base
3. **Top K**: 10
4. **Reranker**: ✅ 勾选
5. **输入问题**: "What are the main issues?"
6. **点击 Search**

应该看到：
- **Summary**: 查询摘要和统计信息
- **Retrieved Documents**: 检索到的文档列表
- **Metadata**: 配置和索引信息

---

## 完成！🎉

你现在有：
- ✅ 固定 URL: `https://你的用户名-rag-search.hf.space`
- ✅ 永久运行（免费 CPU）
- ✅ 自动加载索引和模型
- ✅ 美观的 Gradio 界面

---

## 后续步骤

### 分享你的应用

直接把 URL 发给别人：
```
https://huggingface.co/spaces/你的用户名/rag-search
```

### 添加新模型

1. 训练模型（在 Colab）
2. Push 到 HF Hub:
   ```python
   model.push_to_hub("你的用户名/rag-finetuned-v1")
   ```
3. 修改 `app.py` 第 63 行:
   ```python
   models = {
       'base': SentenceTransformer('Qwen/Qwen3-Embedding-0.6B'),
       'finetuned-v1': SentenceTransformer('你的用户名/rag-finetuned-v1')  # 新增
   }
   ```
4. 提交更新，Space 自动重新构建

### 启用 API（后期对接 Vercel）

你的 Space 已经启用了 API！

访问：`https://你的空间.hf.space/api/`

可以看到自动生成的 API 文档。

后期前端可以直接调用：
```javascript
fetch('https://你的空间.hf.space/api/predict', {
    method: 'POST',
    body: JSON.stringify({
        data: [query, dataset, model, top_k, use_rerank]
    })
})
```

---

## 常见问题

### Q1: 构建失败怎么办？

**检查 Logs**：
- 点击 "Logs" 标签
- 查看错误信息

**常见错误**：
- `YOUR_HF_USERNAME` 没修改 → 修改为你的用户名
- 索引没上传 → 检查 Step 1 是否成功
- Token 权限不足 → 确保 token 是 Write 权限

### Q2: 加载很慢怎么办？

**原因**：
- 首次加载需要下载模型（~2GB）
- 之后会缓存，很快

**解决**：
- 耐心等待首次构建完成
- 后续启动会很快

### Q3: 想要更换数据集怎么办？

1. 在 Colab 上传新索引到 HF Hub
2. Space 会自动检测并加载
3. 无需修改代码

### Q4: 能用 GPU 吗？

可以！
1. 在 Space Settings 中
2. 选择 Hardware: **GPU T4 - Free**（可能需要排队）
3. 保存后自动重启

---

## 资源链接

- **HuggingFace Hub**: https://huggingface.co/
- **你的 Spaces**: https://huggingface.co/spaces
- **你的 Datasets**: https://huggingface.co/datasets
- **Gradio 文档**: https://gradio.app/docs

---

## 需要帮助？

遇到问题？
1. 查看 Space 的 Logs
2. 检查本指南的常见问题
3. 访问 HuggingFace 论坛：https://discuss.huggingface.co/

---

**就是这么简单！** 🚀
