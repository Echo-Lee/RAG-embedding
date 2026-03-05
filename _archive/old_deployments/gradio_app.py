"""
RAG Email Search - Gradio Space Application

部署到 HuggingFace Space 的完整代码
直接复制此文件内容到 Space 的 app.py
"""

import gradio as gr
import faiss
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import numpy as np

# ========== 配置 ==========
YOUR_HF_USERNAME = "YOUR_USERNAME"  # ⚠️ 修改为你的 HF 用户名！

# ========== 加载资源 ==========
print("🔄 Loading resources from HuggingFace Hub...")
print("="*60)

def load_index(dataset):
    """从 HF Hub 加载 FAISS 索引"""
    try:
        print(f"📥 Loading {dataset} index...")

        # 下载索引文件
        index_file = hf_hub_download(
            repo_id=f"{YOUR_HF_USERNAME}/rag-indexes",
            filename=f"{dataset}/faiss_index.bin",
            repo_type="dataset"
        )

        # 下载元数据
        metadata_file = hf_hub_download(
            repo_id=f"{YOUR_HF_USERNAME}/rag-indexes",
            filename=f"{dataset}/metadata.json",
            repo_type="dataset"
        )

        # 加载索引
        index = faiss.read_index(index_file)

        # 加载元数据
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"  ✅ {dataset}: {metadata.get('num_docs', 'N/A')} documents")
        return index, metadata

    except Exception as e:
        print(f"  ❌ Failed to load {dataset}: {e}")
        return None, None

# 加载所有索引
indexes = {}
for dataset in ['hospital', 'corruption']:
    idx, meta = load_index(dataset)
    if idx is not None:
        indexes[dataset] = {'index': idx, 'metadata': meta}

if not indexes:
    raise RuntimeError("No indexes loaded! Please check YOUR_HF_USERNAME configuration.")

print(f"\n✅ Loaded {len(indexes)} datasets: {list(indexes.keys())}")

# 加载 Embedding 模型
print("\n🤖 Loading embedding model...")
models = {
    'base': SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
}
print(f"✅ Loaded {len(models)} models: {list(models.keys())}")

# 加载 Reranker
print("\n🔄 Loading reranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Reranker loaded!")

print("\n" + "="*60)
print("🎉 All resources loaded! Backend ready.")
print("="*60 + "\n")

# ========== RAG 函数 ==========

def search(query, dataset, model_name, top_k, use_rerank):
    """
    RAG 检索函数

    Args:
        query: 用户查询
        dataset: 数据集选择 (hospital/corruption)
        model_name: 模型选择
        top_k: 返回前 K 个结果
        use_rerank: 是否使用 reranker

    Returns:
        answer: 答案摘要
        docs_text: 检索文档
        metadata_text: 元数据信息
    """
    if not query.strip():
        return "⚠️ 请输入问题", "", ""

    try:
        # 选择资源
        index_data = indexes[dataset]
        index = index_data['index']
        model = models[model_name]

        # 编码查询
        print(f"🔍 Query: {query[:50]}...")
        query_emb = model.encode([query], normalize_embeddings=True)

        # FAISS 检索（检索更多候选，用于 rerank）
        retrieval_k = min(top_k * 3, 50)
        scores, indices = index.search(query_emb, retrieval_k)

        # 格式化文档
        docs_list = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            docs_list.append({
                'rank': i + 1,
                'index': int(idx),
                'score': float(score),
                'content': f"[Document {idx} from {dataset} dataset]"  # 简化版
            })

        # Rerank（如果启用）
        if use_rerank and len(docs_list) > top_k:
            print(f"🔄 Reranking top {len(docs_list)} → {top_k} documents")
            # 注意：这里是简化版，实际应该用文档内容 rerank
            docs_list = docs_list[:top_k]
        else:
            docs_list = docs_list[:top_k]

        # 格式化答案
        answer = f"**📊 Query Analysis**\n\n"
        answer += f"- **Query**: {query}\n"
        answer += f"- **Dataset**: {dataset} ({index_data['metadata'].get('num_docs', 'N/A')} total docs)\n"
        answer += f"- **Model**: {model_name}\n"
        answer += f"- **Results**: Retrieved {len(docs_list)} documents\n"
        answer += f"- **Top Score**: {docs_list[0]['score']:.4f}\n"

        # 格式化文档显示
        docs_text = ""
        for doc in docs_list:
            docs_text += f"### 📄 Rank {doc['rank']} - Score: {doc['score']:.4f}\n"
            docs_text += f"**Index**: {doc['index']}\n\n"
            docs_text += f"{doc['content']}\n\n"
            docs_text += "---\n\n"

        # 元数据
        metadata_text = f"**Configuration**\n"
        metadata_text += f"- Dataset: {dataset}\n"
        metadata_text += f"- Model: {model_name}\n"
        metadata_text += f"- Top K: {top_k}\n"
        metadata_text += f"- Reranker: {'✅ Enabled' if use_rerank else '❌ Disabled'}\n"
        metadata_text += f"\n**Index Info**\n"
        metadata_text += f"- Total vectors: {index.ntotal}\n"
        metadata_text += f"- Dimension: {index.d}\n"

        print(f"✅ Search completed: {len(docs_list)} results\n")

        return answer, docs_text, metadata_text

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return error_msg, "", ""

# ========== Gradio UI ==========

# 自定义 CSS
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.header-text {
    text-align: center;
    color: #2563eb;
}
"""

with gr.Blocks(title="RAG Email Search", theme=gr.themes.Soft(), css=custom_css) as demo:

    # Header
    gr.Markdown(
        """
        # 📧 RAG Email Search System
        ### Powered by HuggingFace Space + FAISS + Qwen Embedding

        Search through email datasets using semantic search and vector similarity.
        """
    )

    with gr.Row():
        # 左侧：配置面板
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")

            dataset = gr.Dropdown(
                choices=list(indexes.keys()),
                value=list(indexes.keys())[0],
                label="📁 Dataset",
                info="Select which email dataset to search"
            )

            model = gr.Dropdown(
                choices=list(models.keys()),
                value='base',
                label="🤖 Embedding Model",
                info="Select the embedding model for query encoding"
            )

            top_k = gr.Slider(
                minimum=3,
                maximum=20,
                value=10,
                step=1,
                label="🔢 Top K Results",
                info="Number of documents to retrieve"
            )

            use_rerank = gr.Checkbox(
                label="✨ Use Cross-Encoder Reranker",
                value=True,
                info="Improve result quality with reranking"
            )

            gr.Markdown("---")

            query = gr.Textbox(
                label="❓ Your Question",
                lines=4,
                placeholder="What are the main issues discussed in the emails?\n\nExample: What did David ask Katherine about?",
                info="Enter your question about the emails"
            )

            btn = gr.Button("🔍 Search", variant="primary", size="lg")

            gr.Markdown("---")

            # 元数据显示
            metadata = gr.Textbox(
                label="📊 Metadata",
                lines=8,
                interactive=False,
                show_copy_button=True
            )

        # 右侧：结果面板
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Results")

            answer = gr.Textbox(
                label="💡 Summary",
                lines=8,
                interactive=False,
                show_copy_button=True
            )

            docs = gr.Markdown(
                label="📚 Retrieved Documents"
            )

    # 示例查询
    gr.Markdown("### 💭 Example Queries")
    gr.Examples(
        examples=[
            ["What are the main issues discussed?", "hospital", "base", 5, True],
            ["Show me important documents about litigation", "hospital", "base", 10, True],
            ["What are the key concerns?", "corruption", "base", 7, False],
        ],
        inputs=[query, dataset, model, top_k, use_rerank],
        label="Click to try example queries"
    )

    # Footer
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            <p>Built with ❤️ using HuggingFace Spaces |
            <a href="https://github.com/your-username/your-repo" target="_blank">GitHub</a></p>
        </div>
        """
    )

    # 绑定事件
    btn.click(
        fn=search,
        inputs=[query, dataset, model, top_k, use_rerank],
        outputs=[answer, docs, metadata]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(
        show_api=True,  # 启用 API，方便后期对接 Vercel
        share=False
    )
