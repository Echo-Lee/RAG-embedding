# Quick Start Guide

## 🎯 Goal

Build a complete RAG system with:
- ✅ Dense retrieval (FAISS + Qwen embeddings)
- ✅ Reranking (Cross-Encoder)
- ✅ Answer generation (Azure OpenAI GPT-4)
- ✅ Interactive web UI (Gradio)

## 📦 What's Ready

### Data (Already in Place)
- ✅ Hospital dataset: `data/processed/hospital/threads_with_summary.json` (~9,300 emails, preprocessed)
- ✅ Corruption dataset: `data/processed/corruption/emails_group_by_thread.json` (~800 threads, preprocessed)

### Code (Fully Implemented)
- ✅ Configuration system with YAML
- ✅ Data loader (handles both formats)
- ✅ FAISS indexer (vectorization)
- ✅ Retriever + Reranker (2-stage retrieval)
- ✅ RAG generator (Azure OpenAI)
- ✅ Gradio interface

### Launcher (notebooks/launcher.ipynb)
- ✅ Quick Start mode (load existing index)
- ✅ Full Pipeline mode (build from scratch)
- ⏳ Fine-tune mode (TODO)

## 🚀 How to Use

### Step 1: Open Launcher

Since you use **local Colab connection**:

1. Open `notebooks/launcher.ipynb` in your Colab environment
2. All project files are accessible via local filesystem
3. No need to mount Google Drive

### Step 2: First Time Setup

Set these variables in cell:

```python
MODE = "full"        # Build index from scratch
DATASET = "hospital" # Start with hospital dataset
```

Run all cells. This will:
1. ✅ Install dependencies (~2 min)
2. ✅ Load embedding model (~1 min)
3. ✅ Load 9,300 documents
4. ✅ Encode all documents (~15-20 min on T4 GPU)
5. ✅ Build and save FAISS index
6. ✅ Launch Gradio demo with public URL

### Step 3: Test the System

In the Gradio interface:
1. Enter a question like: "What did David request from Katherine?"
2. Toggle "Use Reranker" to compare results
3. Adjust "Number of Results" (3-20)
4. See retrieved documents and generated answer

### Step 4: Build Corruption Index

Change to:

```python
MODE = "full"
DATASET = "corruption"
```

Run again to build index for the second dataset (~2 min).

### Step 5: Future Sessions (Quick Start)

After indexes are built:

```python
MODE = "quick"       # Fast loading
DATASET = "hospital" # or "corruption"
```

This loads existing index in ~10 seconds and launches demo immediately.

## 📊 What Gets Created

After running Full Pipeline:

```
outputs/indexes/
├── hospital/
│   ├── faiss.index            # ~190 MB vector index
│   ├── doc_metadata.json      # Document metadata
│   └── config.json            # Config cache
└── corruption/
    ├── faiss.index            # ~18 MB vector index
    ├── doc_metadata.json
    └── config.json
```

These files are reused in Quick Start mode.

## 🎛️ Configuration

Edit `experiments/hospital_base.yaml` or `corruption_base.yaml` to adjust:

```yaml
# Retrieval settings
top_k_retrieval: 50    # Candidates for reranking
top_k_rerank: 10       # Final results

# Model settings
embedding_model: Qwen/Qwen3-Embedding-0.6B
reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

# Generation settings
generation_temperature: 0.3
generation_max_tokens: 2000
```

## 🔍 Testing Retrieval Quality

The notebook includes a manual testing cell:

```python
test_query = "What did David R. Park request from Katherine?"

# Retrieve
docs = retriever.retrieve(test_query, top_k=5, use_rerank=True)

# Generate
answer = generator.generate(test_query, docs)
```

Compare results with/without reranker:
- `use_rerank=True` → better accuracy, slightly slower
- `use_rerank=False` → faster, good baseline

## 🎨 Gradio Interface Features

- **Question Input**: Free-text query
- **Settings Panel**:
  - Toggle reranker on/off
  - Adjust top-K results (3-20)
- **Outputs**:
  - Generated answer
  - Retrieved documents with scores
  - Metadata (thread IDs, scores)
- **Examples**: Pre-populated example queries
- **Public URL**: Share with others via `share=True`

## 🔄 Typical Workflow

### First Time (Per Dataset)
```
Open launcher → MODE="full" → Run all cells → Wait 15-20 min → Test demo
```

### Daily Usage
```
Open launcher → MODE="quick" → Run all cells → Demo ready in 10 sec
```

### Experiments
```
Edit YAML config → MODE="full" → Rebuild index → Compare results
```

## 📈 Performance Expectations

### Hospital Dataset (9,300 docs)
- Encoding time: ~15-20 min (T4 GPU)
- Index size: ~190 MB
- Retrieval latency: <100 ms
- With reranker: <200 ms

### Corruption Dataset (800 docs)
- Encoding time: ~2 min (T4 GPU)
- Index size: ~18 MB
- Retrieval latency: <50 ms
- With reranker: <100 ms

## 🐛 Troubleshooting

### "Index not found"
→ Run MODE="full" first to build it

### "CUDA out of memory"
→ Edit config: `batch_size: 16` (default 32)

### "Module not found"
→ First cell installs deps, wait for it to complete

### Slow encoding
→ Check: `print(model.device)` should be "cuda"

### Gradio won't launch
→ Check port 7860 is available

## ✅ Success Checklist

After first run, you should have:
- [ ] Both indexes built (hospital + corruption)
- [ ] Gradio demo launches successfully
- [ ] Can retrieve relevant documents
- [ ] Can generate reasonable answers
- [ ] Reranker improves top results
- [ ] Public URL works for sharing

## 🎯 Next Steps

Once basic pipeline works:
1. Experiment with different reranker models
2. Tune retrieval parameters (top_k values)
3. Implement fine-tuning for custom embeddings
4. Add evaluation metrics (Hit@K, MRR)
5. Deploy as standalone app

---

**You're all set!** Open `notebooks/launcher.ipynb` and start building indexes. 🚀
