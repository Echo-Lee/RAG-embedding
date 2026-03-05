# RAG Email System

Retrieval-Augmented Generation (RAG) system for querying email datasets with semantic search and LLM generation.

**🎯 Three-Step Workflow**: Train → Evaluate → Deploy

---

## 🚀 Quick Start

### **Step 1: Train Models** (~30-60 mins)

1. Open **[colab_train.ipynb](colab_train.ipynb)** in Google Colab
2. Click Runtime → Run all
3. Enter HuggingFace token when prompted
4. Wait for training to complete

**Result**: Base + Fine-tuned models, saved to Drive and HuggingFace Hub

### **Step 2: Evaluate Performance** (~10-15 mins)

1. Open **[colab_evaluation.ipynb](colab_evaluation.ipynb)** in Google Colab
2. Run all cells
3. View performance metrics and visualizations

**Result**: Detailed analysis with charts and improvement metrics

### **Step 3: Deploy to HuggingFace Space** (~5 mins)

1. Create Space at https://huggingface.co/spaces
2. Copy **[app.py](app.py)** contents
3. Deploy and get permanent URL

**Result**: Public web app with dual-model comparison

---

## 📋 What It Does

- **Training**: Fine-tunes embedding models using LoRA on email data
- **Retrieval**: FAISS search with optional cross-encoder reranking
- **Comparison**: Side-by-side base vs fine-tuned model evaluation
- **Deployment**: Web interface with model comparison on HuggingFace Spaces
- **Persistence**: Auto-saves to Google Drive and HuggingFace Hub

---

## 📁 Project Structure

```
RAG-embedding/
├── 🚀 Main Workflows
│   ├── colab_train.ipynb      # ⭐ One-click training pipeline
│   ├── colab_evaluation.ipynb # ⭐ Model evaluation & analysis
│   └── app.py                 # ⭐ HuggingFace Space deployment
│
├── src/                        # Core modules
│   ├── config/                 # Configuration management
│   ├── data/                   # Data loaders
│   ├── retrieval/              # FAISS + reranking
│   ├── generation/             # Azure OpenAI integration
│   └── app/                    # Gradio UI components
│
├── experiments/                # YAML configs
│   ├── hospital_base_template.yaml
│   └── corruption_base_template.yaml
│
├── _archive/                   # Legacy files (reference only)
│   ├── old_notebooks/
│   ├── old_deployments/
│   └── old_guides/
│
├── CLAUDE.md                   # Detailed documentation
├── README.md                   # This file
└── requirements.txt            # Dependencies

Google Drive (auto-created by training):
└── Epiq Project/pipeline/
    ├── data/processed/         # Your email data
    ├── faiss_index/            # Built indexes (base + fine-tuned)
    ├── models/                 # Fine-tuned models
    └── evaluation_results/     # Analysis outputs

HuggingFace Hub (auto-uploaded by training):
├── ChenyuEcho/rag-indexes                  # All FAISS indexes
├── ChenyuEcho/rag-finetuned-hospital       # Fine-tuned model
└── ChenyuEcho/rag-finetuned-corruption     # Fine-tuned model
```

---

## 🛠️ Prerequisites

### 1. Upload Data to Google Drive

Place your email data at:
```
/content/drive/MyDrive/Epiq Project/pipeline/data/processed/
├── hospital/threads_with_summary.json      (28MB, ~9,300 emails)
└── corruption/emails_group_by_thread.json  (3.3MB, ~800 threads)
```

### 2. Get HuggingFace Token

1. Visit https://huggingface.co/settings/tokens
2. Create token with **Write** permission
3. Save for training step

---

## 📖 Detailed Usage

### Training (colab_train.ipynb)

**What it does**:
1. Mounts Google Drive
2. Pulls latest code from GitHub
3. Loads your email data
4. Builds base model indexes (Qwen3-Embedding-0.6B)
5. Fine-tunes models using LoRA
6. Builds fine-tuned model indexes
7. Uploads everything to HuggingFace Hub

**Configuration** (in notebook):
```python
# Which datasets to train
TRAIN_DATASETS = ['hospital', 'corruption']  # or just ['hospital']

# Training parameters
TRAIN_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
```

### Evaluation (colab_evaluation.ipynb)

**What it analyzes**:
- Top-1 and average retrieval scores
- Score improvements (base vs fine-tuned)
- Embedding quality (PCA, norms, variance)
- Per-query performance breakdown

**Configuration**:
```python
# Choose dataset to evaluate
EVAL_DATASET = "hospital"  # or "corruption"
```

### Deployment (app.py)

**Features**:
- Single model search tab
- Dual-model comparison tab
- Real-time retrieval
- Top-K control (5-20)
- Improvement metrics

**Configuration**:
- Automatically loads from `ChenyuEcho/rag-indexes`
- Supports both datasets
- Auto-detects available models

---

## 🔄 Local Development

### Edit Code Locally, Run on Colab GPU

1. **Edit locally**:
   ```bash
   # Edit src/retrieval/retriever.py
   git add . && git commit -m "Update" && git push
   ```

2. **Update Colab**:
   - Re-run first cell (auto `git pull`)
   - Or manually: `!cd /content/RAG-embedding && git pull`

3. **Restart kernel**, re-run notebook

---

## ⚙️ Configuration

Example `experiments/hospital_base.yaml`:

```yaml
dataset:
  name: hospital
  data_path: data/processed/hospital/threads_with_summary.json

embedding_model: Qwen/Qwen3-Embedding-0.6B
top_k_retrieval: 50
top_k_rerank: 10
use_reranker: true

azure_api_key: YOUR_KEY
azure_endpoint: YOUR_ENDPOINT
azure_deployment: gpt-4
```

---

## 📊 Performance Metrics

### Training Time (with T4 GPU)

| Task | Hospital | Corruption | Notes |
|------|----------|------------|-------|
| **Build base index** | ~15 min | ~2 min | 9,300 vs 800 docs |
| **Fine-tune model (LoRA)** | ~10 min | ~5 min | 3 epochs |
| **Build fine-tuned index** | ~15 min | ~2 min | Same as base |
| **Upload to HF Hub** | ~5 min | ~2 min | Depends on network |
| **Total** | ~45 min | ~11 min | End-to-end |

### Evaluation Time

| Task | Time | Notes |
|------|------|-------|
| **Load models/indexes** | ~2 min | From Drive |
| **Run test queries** | ~1 min | 8 queries × 2 models |
| **Generate visualizations** | ~2 min | 4 plots + PCA |
| **Total** | ~5 min | Complete analysis |

### Expected Improvements

Based on fine-tuning with LoRA:
- **Top-1 score**: +15-30% improvement
- **Average score**: +20-35% improvement
- **Domain-specific queries**: Best improvements

---

## 🎯 Tips

- **First time**: Run training with both datasets to compare
- **Colab disconnects**: All outputs saved to Drive, safe to restart
- **GPU usage**: Training uses GPU, evaluation can use CPU
- **HF Space**: Free tier is sufficient for deployment

---

## 🐛 Troubleshooting

### Training Issues

**"Permission denied" when uploading to HF Hub**
- Ensure token has **Write** permission
- Regenerate token if needed

**"CUDA out of memory"**
- Reduce `TRAIN_BATCH_SIZE` (default: 8)
- Reduce `BATCH_SIZE` (default: 32)

**"Data not found"**
- Verify Drive path: `/content/drive/MyDrive/Epiq Project/pipeline/data/processed/`
- Check file names match exactly

### Evaluation Issues

**"Index not found"**
- Run training first to generate indexes
- Check Drive path: `~/Epiq Project/pipeline/faiss_index/`

**"Model loading fails"**
- Ensure training completed successfully
- Check Drive path: `~/Epiq Project/pipeline/models/`

### Deployment Issues

**"Failed to load from HF Hub"**
- Verify repositories exist: `ChenyuEcho/rag-indexes`
- Check repository visibility (should be public)
- Wait a few minutes after upload

**Space build fails**
- Check `app.py` syntax
- Verify `HF_USERNAME = "ChenyuEcho"` is correct
- Check Space logs for detailed errors

---

## 📝 Tech Stack

### Models & Libraries
- **Embeddings**: Qwen3-Embedding-0.6B (base model)
- **Fine-tuning**: LoRA (Parameter-Efficient Fine-Tuning)
- **Vector Search**: FAISS (IndexFlatIP for cosine similarity)
- **Reranking**: Cross-Encoder (optional, ms-marco-MiniLM)
- **Generation**: Azure OpenAI GPT-4.1 (for Q&A)

### Infrastructure
- **Training**: Google Colab (T4/V100 GPU)
- **Storage**: Google Drive (persistent storage)
- **Deployment**: HuggingFace Spaces (free tier)
- **UI**: Gradio (web interface)

### Data
- **Hospital**: ~9,300 emails, 28MB
- **Corruption**: ~800 emails, 3.3MB
- **Format**: Thread-based JSON with metadata

---

## 📚 Key Code Modules

| Module | Purpose |
|--------|---------|
| [src/config/config.py](src/config/config.py) | Configuration management (YAML → dataclass) |
| [src/data/loader.py](src/data/loader.py) | Multi-format email data loading |
| [src/retrieval/indexer.py](src/retrieval/indexer.py) | FAISS index building and saving |
| [src/retrieval/retriever.py](src/retrieval/retriever.py) | Hybrid retrieval (FAISS + reranking) |
| [src/retrieval/reranker.py](src/retrieval/reranker.py) | Cross-encoder reranking |
| [src/generation/rag_generator.py](src/generation/rag_generator.py) | Azure OpenAI integration |
| [src/app/gradio_app.py](src/app/gradio_app.py) | Gradio UI components |

---

## 🔗 Links

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed guide
- **GitHub**: https://github.com/Echo-Lee/RAG-embedding
- **HuggingFace**: https://huggingface.co/ChenyuEcho
- **Colab**: Open notebooks directly in Google Colab

---

## 📄 License

Academic project - Cornell University

---

**Built for efficient email search and analysis** 🎓
