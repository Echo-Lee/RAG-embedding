"""
RAG Email Search - Gradio Space Application (Model Comparison Version)

支持双栏对比：微调前 vs 微调后
部署到 HuggingFace Space 的完整代码
"""

import gradio as gr
import faiss
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import numpy as np

# ========== 配置 ==========
YOUR_HF_USERNAME = "YOUR_USERNAME"  # ⚠️ 修改为你的 HF 用户名！

# 模型配置（如果有 fine-tuned 模型，在这里添加）
MODEL_CONFIGS = {
    'base': {
        'name': 'Base Model',
        'model_id': 'Qwen/Qwen3-Embedding-0.6B',
        'description': 'Pre-trained Qwen3 embedding model'
    },
    # 如果有 fine-tuned 模型，取消注释并修改：
    # 'finetuned-hospital': {
    #     'name': 'Fine-tuned (Hospital)',
    #     'model_id': f'{YOUR_HF_USERNAME}/rag-finetuned-hospital',
    #     'description': 'Fine-tuned on hospital emails'
    # },
}

# ========== 加载资源 ==========
print("🔄 Loading resources from HuggingFace Hub...")
print("="*60)

def load_index(dataset):
    """从 HF Hub 加载 FAISS 索引"""
    try:
        print(f"📥 Loading {dataset} index...")

        index_file = hf_hub_download(
            repo_id=f"{YOUR_HF_USERNAME}/rag-indexes",
            filename=f"{dataset}/faiss_index.bin",
            repo_type="dataset"
        )

        metadata_file = hf_hub_download(
            repo_id=f"{YOUR_HF_USERNAME}/rag-indexes",
            filename=f"{dataset}/metadata.json",
            repo_type="dataset"
        )

        index = faiss.read_index(index_file)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"  ✅ {dataset}: {metadata.get('num_docs', 'N/A')} documents")
        return index, metadata

    except Exception as e:
        print(f"  ❌ Failed to load {dataset}: {e}")
        return None, None

# 加载索引
indexes = {}
for dataset in ['hospital', 'corruption']:
    idx, meta = load_index(dataset)
    if idx is not None:
        indexes[dataset] = {'index': idx, 'metadata': meta}

if not indexes:
    raise RuntimeError("No indexes loaded! Please check YOUR_HF_USERNAME configuration.")

print(f"\n✅ Loaded {len(indexes)} datasets: {list(indexes.keys())}")

# 加载所有模型
print("\n🤖 Loading embedding models...")
models = {}
for model_key, model_config in MODEL_CONFIGS.items():
    try:
        print(f"  Loading {model_config['name']}...")
        models[model_key] = SentenceTransformer(model_config['model_id'])
        print(f"    ✅ Loaded")
    except Exception as e:
        print(f"    ❌ Failed: {e}")

if not models:
    raise RuntimeError("No models loaded!")

print(f"✅ Loaded {len(models)} models: {list(models.keys())}")

# 加载 Reranker
print("\n🔄 Loading reranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Reranker loaded!")

print("\n" + "="*60)
print("🎉 All resources loaded! Backend ready.")
print("="*60 + "\n")

# ========== 检索函数 ==========

def retrieve_with_model(query, dataset, model_key, top_k, use_rerank):
    """使用指定模型检索"""
    index_data = indexes[dataset]
    index = index_data['index']
    model = models[model_key]

    # 编码查询
    query_emb = model.encode([query], normalize_embeddings=True)

    # FAISS 检索
    retrieval_k = min(top_k * 3, 50)
    scores, indices = index.search(query_emb, retrieval_k)

    # 格式化文档
    docs_list = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        docs_list.append({
            'rank': i + 1,
            'index': int(idx),
            'score': float(score),
            'content': f"[Document {idx}]"  # 简化版
        })

    # Rerank（可选）
    if use_rerank and len(docs_list) > top_k:
        # 简化版
        docs_list = docs_list[:top_k]
    else:
        docs_list = docs_list[:top_k]

    return docs_list

# ========== UI 函数 ==========

def search_single(query, dataset, model_key, top_k, use_rerank):
    """单模型搜索"""
    if not query.strip():
        return "⚠️ 请输入问题", "", ""

    try:
        docs = retrieve_with_model(query, dataset, model_key, top_k, use_rerank)

        # 格式化答案
        model_name = MODEL_CONFIGS[model_key]['name']
        answer = f"**Model**: {model_name}\n"
        answer += f"**Dataset**: {dataset}\n"
        answer += f"**Results**: {len(docs)} documents\n"
        answer += f"**Top Score**: {docs[0]['score']:.4f}\n"

        # 格式化文档
        docs_text = ""
        for doc in docs:
            docs_text += f"### Rank {doc['rank']} - Score: {doc['score']:.4f}\n"
            docs_text += f"Index: {doc['index']}\n\n"
            docs_text += f"{doc['content']}\n\n---\n\n"

        # 元数据
        metadata = f"Model: {model_name}\nDataset: {dataset}\nTop K: {top_k}\nReranker: {use_rerank}"

        return answer, docs_text, metadata

    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

def compare_models(query, dataset, model1_key, model2_key, top_k, use_rerank):
    """对比两个模型"""
    if not query.strip():
        return "⚠️ 请输入问题", "", "⚠️ 请输入问题", ""

    try:
        # 检索两个模型
        docs1 = retrieve_with_model(query, dataset, model1_key, top_k, use_rerank)
        docs2 = retrieve_with_model(query, dataset, model2_key, top_k, use_rerank)

        # 格式化模型 1 结果
        model1_name = MODEL_CONFIGS[model1_key]['name']
        answer1 = f"**Model**: {model1_name}\n"
        answer1 += f"**Top Score**: {docs1[0]['score']:.4f}\n"
        answer1 += f"**Avg Score**: {np.mean([d['score'] for d in docs1]):.4f}\n"

        docs1_text = ""
        for doc in docs1:
            docs1_text += f"**{doc['rank']}. Score: {doc['score']:.4f}**\n"
            docs1_text += f"Index: {doc['index']}\n\n---\n\n"

        # 格式化模型 2 结果
        model2_name = MODEL_CONFIGS[model2_key]['name']
        answer2 = f"**Model**: {model2_name}\n"
        answer2 += f"**Top Score**: {docs2[0]['score']:.4f}\n"
        answer2 += f"**Avg Score**: {np.mean([d['score'] for d in docs2]):.4f}\n"

        docs2_text = ""
        for doc in docs2:
            docs2_text += f"**{doc['rank']}. Score: {doc['score']:.4f}**\n"
            docs2_text += f"Index: {doc['index']}\n\n---\n\n"

        return answer1, docs1_text, answer2, docs2_text

    except Exception as e:
        error = f"❌ Error: {str(e)}"
        return error, "", error, ""

# ========== Gradio UI ==========

custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.compare-header {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    color: white;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    margin-bottom: 10px;
}
"""

with gr.Blocks(title="RAG Model Comparison", theme=gr.themes.Soft(), css=custom_css) as demo:

    gr.Markdown(
        """
        # 📊 RAG Model Comparison System
        ### Compare Base Model vs Fine-tuned Model Side-by-Side

        Compare retrieval quality between different embedding models.
        """
    )

    # 模式选择
    mode = gr.Radio(
        choices=["Single Model", "Compare Models"],
        value="Single Model" if len(models) == 1 else "Compare Models",
        label="Mode",
        info="Choose single model or side-by-side comparison"
    )

    # 共享配置
    with gr.Row():
        dataset = gr.Dropdown(
            choices=list(indexes.keys()),
            value=list(indexes.keys())[0],
            label="📁 Dataset"
        )
        top_k = gr.Slider(3, 20, 10, step=1, label="🔢 Top K")
        use_rerank = gr.Checkbox(label="✨ Reranker", value=True)

    query = gr.Textbox(
        label="❓ Your Question",
        lines=3,
        placeholder="What are the main issues discussed?"
    )

    # === Single Model Mode ===
    with gr.Group(visible=True) as single_mode:
        gr.Markdown("### Single Model Search")

        model_single = gr.Dropdown(
            choices=list(models.keys()),
            value=list(models.keys())[0],
            label="🤖 Model"
        )

        btn_single = gr.Button("🔍 Search", variant="primary", size="lg")

        with gr.Row():
            answer_single = gr.Textbox(label="Summary", lines=5)
            metadata_single = gr.Textbox(label="Metadata", lines=5)

        docs_single = gr.Markdown(label="Retrieved Documents")

    # === Compare Mode ===
    with gr.Group(visible=(len(models) > 1)) as compare_mode:
        gr.Markdown("### Model Comparison")

        with gr.Row():
            model1 = gr.Dropdown(
                choices=list(models.keys()),
                value=list(models.keys())[0],
                label="🤖 Model 1 (Left)"
            )
            model2 = gr.Dropdown(
                choices=list(models.keys())[1:] if len(models) > 1 else list(models.keys()),
                value=list(models.keys())[1] if len(models) > 1 else list(models.keys())[0],
                label="🤖 Model 2 (Right)"
            )

        btn_compare = gr.Button("⚖️ Compare", variant="primary", size="lg")

        with gr.Row():
            # 左侧：模型 1
            with gr.Column():
                gr.HTML("<div class='compare-header'>Model 1</div>")
                answer1 = gr.Textbox(label="Summary", lines=5)
                docs1 = gr.Markdown(label="Top Results")

            # 右侧：模型 2
            with gr.Column():
                gr.HTML("<div class='compare-header'>Model 2</div>")
                answer2 = gr.Textbox(label="Summary", lines=5)
                docs2 = gr.Markdown(label="Top Results")

    # 示例
    gr.Examples(
        examples=[
            ["What are the main issues?", "hospital", 5, True],
            ["Show me important documents", "corruption", 10, True],
        ],
        inputs=[query, dataset, top_k, use_rerank]
    )

    # 模式切换逻辑
    def switch_mode(mode_choice):
        if mode_choice == "Single Model":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    mode.change(
        fn=switch_mode,
        inputs=[mode],
        outputs=[single_mode, compare_mode]
    )

    # 事件绑定
    btn_single.click(
        fn=search_single,
        inputs=[query, dataset, model_single, top_k, use_rerank],
        outputs=[answer_single, docs_single, metadata_single]
    )

    btn_compare.click(
        fn=compare_models,
        inputs=[query, dataset, model1, model2, top_k, use_rerank],
        outputs=[answer1, docs1, answer2, docs2]
    )

    # Footer
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #666;">
            <p>Built with ❤️ using HuggingFace Spaces | Model Comparison Mode</p>
        </div>
        """
    )

# 启动
if __name__ == "__main__":
    demo.launch(show_api=True, share=False)
