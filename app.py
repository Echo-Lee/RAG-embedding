"""
RAG Email Search - HuggingFace Space App (Model Comparison)

双栏对比：Base Model vs Fine-tuned Model
"""

import gradio as gr
import faiss
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from pathlib import Path

# ========== 配置 ==========
HF_USERNAME = "ChenyuEcho"
HF_INDEX_REPO = f"{HF_USERNAME}/rag-indexes"

# 模型配置
MODEL_CONFIGS = {
    'base': {
        'name': 'Base Model',
        'model_id': 'Qwen/Qwen3-Embedding-0.6B',
        'description': 'Pre-trained Qwen3 embedding model (no fine-tuning)'
    },
    'finetuned-hospital': {
        'name': 'Fine-tuned (Hospital)',
        'model_id': f'{HF_USERNAME}/rag-finetuned-hospital',
        'description': 'Fine-tuned on hospital email dataset'
    },
    'finetuned-corruption': {
        'name': 'Fine-tuned (Corruption)',
        'model_id': f'{HF_USERNAME}/rag-finetuned-corruption',
        'description': 'Fine-tuned on corruption email dataset'
    }
}

# 数据集配置
DATASETS = ['hospital', 'corruption']

print("="*60)
print("🚀 RAG Email Search - HuggingFace Space")
print("="*60)

# ========== 加载资源 ==========
print("\n📥 Loading resources from HuggingFace Hub...\n")

# 存储资源
indexes = {}
doc_metadata = {}
models = {}

def load_index(dataset, model_type):
    """
    从 HF Hub 加载 FAISS 索引

    Args:
        dataset: 'hospital' or 'corruption'
        model_type: 'base' or 'finetuned'
    """
    index_name = f"{model_type}-{dataset}"

    try:
        print(f"  📥 Loading index: {index_name}...")

        # 下载索引文件
        index_file = hf_hub_download(
            repo_id=HF_INDEX_REPO,
            filename=f"{index_name}/faiss_index.bin",
            repo_type="dataset"
        )

        metadata_file = hf_hub_download(
            repo_id=HF_INDEX_REPO,
            filename=f"{index_name}/metadata.json",
            repo_type="dataset"
        )

        doc_meta_file = hf_hub_download(
            repo_id=HF_INDEX_REPO,
            filename=f"{index_name}/doc_metadata.json",
            repo_type="dataset"
        )

        # 加载索引
        index = faiss.read_index(index_file)

        # 加载元数据
        with open(metadata_file, 'r') as f:
            meta = json.load(f)

        with open(doc_meta_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)

        print(f"    ✅ {meta['num_docs']} documents")

        return index, docs

    except Exception as e:
        print(f"    ❌ Error loading {index_name}: {e}")
        return None, None

# 加载所有索引
for dataset in DATASETS:
    indexes[f'base-{dataset}'] = {}
    indexes[f'finetuned-{dataset}'] = {}
    doc_metadata[dataset] = None

    # Base index
    idx, docs = load_index(dataset, 'base')
    if idx is not None:
        indexes[f'base-{dataset}']['index'] = idx
        doc_metadata[dataset] = docs

    # Fine-tuned index
    idx, _ = load_index(dataset, 'finetuned')  # 使用相同的doc_metadata
    if idx is not None:
        indexes[f'finetuned-{dataset}']['index'] = idx

print("\n🤖 Loading models...\n")

# 加载模型
for model_key, model_config in MODEL_CONFIGS.items():
    try:
        print(f"  📥 Loading: {model_config['name']}...")
        model = SentenceTransformer(model_config['model_id'])
        models[model_key] = model
        print(f"    ✅ Loaded")
    except Exception as e:
        print(f"    ⚠️  Failed to load {model_key}: {e}")

print(f"\n✅ Loaded {len(models)} models")
print("="*60)

# ========== 检索函数 ==========

def retrieve(query, dataset, model_key, top_k=10):
    """
    检索函数

    Args:
        query: 查询文本
        dataset: 数据集名称
        model_key: 模型键（'base', 'finetuned-hospital', etc.）
        top_k: 返回前K个结果

    Returns:
        检索结果列表
    """
    # 确定索引类型
    if model_key == 'base':
        index_key = f'base-{dataset}'
    else:
        index_key = f'finetuned-{dataset}'

    # 检查资源是否存在
    if index_key not in indexes or 'index' not in indexes[index_key]:
        return None, f"Index not available: {index_key}"

    if model_key not in models:
        return None, f"Model not available: {model_key}"

    if dataset not in doc_metadata or doc_metadata[dataset] is None:
        return None, f"Document metadata not available: {dataset}"

    # 获取资源
    index = indexes[index_key]['index']
    model = models[model_key]
    docs = doc_metadata[dataset]

    # 编码查询
    query_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)

    # 检索
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_emb, k)

    # 构建结果
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(docs):
            continue

        doc = docs[idx]
        results.append({
            'rank': i + 1,
            'score': float(score),
            'doc_index': int(idx),
            'content': doc['content'],
            'metadata': doc.get('metadata', {})
        })

    return results, None

def format_results(results, model_name):
    """格式化显示结果"""
    if not results:
        return "No results"

    output = f"**Model**: {model_name}\n\n"
    output += f"**Top Score**: {results[0]['score']:.4f}\n"
    output += f"**Avg Score**: {np.mean([r['score'] for r in results]):.4f}\n\n"
    output += "---\n\n"

    for r in results:
        output += f"**{r['rank']}. Score: {r['score']:.4f}** (Index: {r['doc_index']})\n\n"

        # 显示内容前300字符
        content_preview = r['content'][:300]
        if len(r['content']) > 300:
            content_preview += "..."

        output += f"```\n{content_preview}\n```\n\n"
        output += "---\n\n"

    return output

# ========== Gradio 界面 ==========

def search_single(query, dataset, model_key, top_k):
    """单模型检索"""
    if not query or not query.strip():
        return "Please enter a query", ""

    results, error = retrieve(query, dataset, model_key, top_k)

    if error:
        return f"Error: {error}", ""

    model_name = MODEL_CONFIGS.get(model_key, {}).get('name', model_key)
    output = format_results(results, model_name)

    # 元数据
    meta_info = f"**Dataset**: {dataset}\n"
    meta_info += f"**Model**: {model_name}\n"
    meta_info += f"**Results**: {len(results)}"

    return output, meta_info

def search_compare(query, dataset, model1_key, model2_key, top_k):
    """双模型对比"""
    if not query or not query.strip():
        return "Please enter a query", "", ""

    # Model 1
    results1, error1 = retrieve(query, dataset, model1_key, top_k)
    if error1:
        output1 = f"Error: {error1}"
    else:
        model1_name = MODEL_CONFIGS.get(model1_key, {}).get('name', model1_key)
        output1 = format_results(results1, model1_name)

    # Model 2
    results2, error2 = retrieve(query, dataset, model2_key, top_k)
    if error2:
        output2 = f"Error: {error2}"
    else:
        model2_name = MODEL_CONFIGS.get(model2_key, {}).get('name', model2_key)
        output2 = format_results(results2, model2_name)

    # 对比统计
    if not error1 and not error2:
        score1_top = results1[0]['score']
        score2_top = results2[0]['score']
        score1_avg = np.mean([r['score'] for r in results1])
        score2_avg = np.mean([r['score'] for r in results2])

        improvement_top = (score2_top - score1_top) / score1_top * 100
        improvement_avg = (score2_avg - score1_avg) / score1_avg * 100

        comparison = f"""## Comparison Summary

**Dataset**: {dataset}
**Top-K**: {top_k}

| Metric | {MODEL_CONFIGS[model1_key]['name']} | {MODEL_CONFIGS[model2_key]['name']} | Δ |
|--------|----------|----------|---|
| **Top Score** | {score1_top:.4f} | {score2_top:.4f} | {improvement_top:+.1f}% |
| **Avg Score** | {score1_avg:.4f} | {score2_avg:.4f} | {improvement_avg:+.1f}% |
"""
    else:
        comparison = "Comparison not available due to errors"

    return output1, output2, comparison

# Gradio 界面定义
with gr.Blocks(title="RAG Email Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📧 RAG Email Search System

    **Dual-Model Comparison**: Compare Base Model vs Fine-tuned Models

    Choose between single model search or side-by-side comparison.
    """)

    with gr.Tab("🔍 Single Model Search"):
        with gr.Row():
            with gr.Column():
                query_single = gr.Textbox(
                    label="Query",
                    placeholder="Enter your question here...",
                    lines=2
                )
                dataset_single = gr.Dropdown(
                    choices=DATASETS,
                    value=DATASETS[0],
                    label="Dataset"
                )
                model_single = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value='base',
                    label="Model"
                )
                top_k_single = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Top K"
                )
                search_btn_single = gr.Button("🔍 Search", variant="primary")

        with gr.Row():
            with gr.Column():
                results_single = gr.Markdown(label="Results")
            with gr.Column(scale=0.3):
                meta_single = gr.Markdown(label="Metadata")

        search_btn_single.click(
            fn=search_single,
            inputs=[query_single, dataset_single, model_single, top_k_single],
            outputs=[results_single, meta_single]
        )

    with gr.Tab("⚖️ Model Comparison"):
        with gr.Row():
            with gr.Column():
                query_compare = gr.Textbox(
                    label="Query",
                    placeholder="Enter your question here...",
                    lines=2
                )
                dataset_compare = gr.Dropdown(
                    choices=DATASETS,
                    value=DATASETS[0],
                    label="Dataset"
                )
                with gr.Row():
                    model1_compare = gr.Dropdown(
                        choices=list(MODEL_CONFIGS.keys()),
                        value='base',
                        label="Model 1 (Left)"
                    )
                    model2_compare = gr.Dropdown(
                        choices=list(MODEL_CONFIGS.keys()),
                        value='finetuned-hospital',
                        label="Model 2 (Right)"
                    )
                top_k_compare = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Top K"
                )
                search_btn_compare = gr.Button("⚖️ Compare", variant="primary")

        comparison_summary = gr.Markdown(label="Comparison Summary")

        with gr.Row():
            with gr.Column():
                results_model1 = gr.Markdown(label="Model 1 Results")
            with gr.Column():
                results_model2 = gr.Markdown(label="Model 2 Results")

        search_btn_compare.click(
            fn=search_compare,
            inputs=[query_compare, dataset_compare, model1_compare, model2_compare, top_k_compare],
            outputs=[results_model1, results_model2, comparison_summary]
        )

    gr.Markdown("""
    ---

    ### 📊 About

    This application uses a Retrieval-Augmented Generation (RAG) system to search through email datasets.

    **Models**:
    - **Base**: Pre-trained Qwen3 embedding model
    - **Fine-tuned**: Models fine-tuned on specific email datasets

    **Datasets**:
    - **Hospital**: ~9,300 hospital-related emails
    - **Corruption**: ~800 corruption investigation emails

    🔗 [GitHub](https://github.com/Echo-Lee/RAG-embedding) | Built with [Gradio](https://gradio.app)
    """)

# 启动应用
if __name__ == "__main__":
    demo.launch()
