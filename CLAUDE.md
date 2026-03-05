# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Retrieval-Augmented Generation (RAG)** system for email search and analysis. The system uses dense retrieval (FAISS + embeddings), optional cross-encoder reranking, and Azure OpenAI for answer generation.

**Primary Use Case**: Querying email datasets (hospital, corruption) with semantic search and LLM-powered responses.

**Deployment**: Google Colab for development/training, HuggingFace Spaces for production demos.

## 🚀 Recommended Workflow (Updated)

This project now has a streamlined workflow with three main notebooks:

### 1. **Training**: [colab_train.ipynb](colab_train.ipynb)
**One-click training pipeline** that handles everything:
- ✅ Mounts Google Drive at `/content/drive/MyDrive/Epiq Project/pipeline/`
- ✅ Git pulls latest code from `https://github.com/Echo-Lee/RAG-embedding.git`
- ✅ Loads data from Drive (`data/processed/hospital/` and `data/processed/corruption/`)
- ✅ Builds **base model** indexes (using Qwen3-Embedding-0.6B)
- ✅ Fine-tunes models using **LoRA** (anchor-positive pairs from same threads)
- ✅ Builds **fine-tuned model** indexes
- ✅ Saves all outputs to Drive:
  - Indexes: `faiss_index/base-{dataset}/` and `faiss_index/finetuned-{dataset}/`
  - Models: `models/finetuned-{dataset}/`
- ✅ Uploads to HuggingFace Hub:
  - Indexes → `ChenyuEcho/rag-indexes`
  - Models → `ChenyuEcho/rag-finetuned-{dataset}`

**To use**: Open in Colab → Runtime → Run all (takes ~30-60 mins with GPU)

### 2. **Evaluation**: [colab_evaluation.ipynb](colab_evaluation.ipynb)
**Comprehensive model analysis**:
- ✅ Loads base and fine-tuned models/indexes from Drive
- ✅ Runs test queries on both models
- ✅ Compares retrieval quality (Top-1 scores, Avg scores, improvements)
- ✅ Generates visualizations (score distributions, PCA embeddings)
- ✅ Analyzes embedding quality (norms, variance, cosine similarity)
- ✅ Creates markdown evaluation report
- ✅ Saves results to Drive: `evaluation_results/{dataset}/`

**To use**: Open after training → Run all (takes ~10-15 mins)

### 3. **Deployment**: [app.py](app.py)
**HuggingFace Space application** with dual-model comparison:
- ✅ Auto-loads indexes and models from `ChenyuEcho/rag-indexes` and `ChenyuEcho/rag-finetuned-*`
- ✅ Two tabs:
  - **Single Model Search**: Test one model at a time
  - **Model Comparison**: Side-by-side comparison with improvement metrics
- ✅ Supports both `hospital` and `corruption` datasets
- ✅ Real-time retrieval with top-K control

**To deploy**:
1. Create HF Space: https://huggingface.co/spaces
2. Copy [app.py](app.py) contents
3. Space auto-installs dependencies and loads resources

### Google Drive Structure

```
/content/drive/MyDrive/Epiq Project/pipeline/
├── data/processed/
│   ├── hospital/threads_with_summary.json       (your data)
│   └── corruption/emails_group_by_thread.json   (your data)
├── faiss_index/
│   ├── base-hospital/                           (base model index)
│   │   ├── faiss_index.bin
│   │   ├── metadata.json
│   │   └── doc_metadata.json
│   ├── finetuned-hospital/                      (fine-tuned index)
│   │   └── (same structure)
│   ├── base-corruption/
│   └── finetuned-corruption/
├── models/
│   ├── finetuned-hospital/                      (fine-tuned model)
│   └── finetuned-corruption/
└── evaluation_results/
    ├── hospital/
    │   ├── evaluation_report_hospital.md
    │   ├── top1_comparison.png
    │   ├── score_distribution.png
    │   ├── improvement_per_query.png
    │   └── embedding_pca.png
    └── corruption/
        └── (same structure)
```

### HuggingFace Hub Structure

```
ChenyuEcho/
├── rag-indexes (dataset)
│   ├── base-hospital/
│   ├── finetuned-hospital/
│   ├── base-corruption/
│   └── finetuned-corruption/
├── rag-finetuned-hospital (model)
└── rag-finetuned-corruption (model)
```

## Architecture

### Core Pipeline

The RAG pipeline follows this flow:

1. **Data Loading** ([src/data/loader.py](src/data/loader.py))
   - Loads email threads from JSON files
   - Supports two formats: hospital (`threads_with_summary.json`) and corruption (`emails_group_by_thread.json`)
   - Enriches emails with metadata (subject, sender, recipient, date)
   - Returns `EmailDocument` objects

2. **Index Building** ([src/retrieval/indexer.py](src/retrieval/indexer.py))
   - Encodes documents using SentenceTransformer (default: Qwen3-Embedding-0.6B)
   - Creates FAISS index (IndexFlatIP for cosine similarity)
   - Saves index + metadata to disk for reuse

3. **Retrieval** ([src/retrieval/retriever.py](src/retrieval/retriever.py))
   - **Stage 1**: Dense retrieval via FAISS (top-k=50 by default)
   - **Stage 2** (optional): Cross-encoder reranking (narrows to top-k=10)
   - Returns ranked documents with scores

4. **Generation** ([src/generation/rag_generator.py](src/generation/rag_generator.py))
   - Takes query + retrieved documents
   - Calls Azure OpenAI GPT-4 to generate answer
   - Returns answer with optional source citations

### Configuration System

All parameters are managed via YAML configs in [experiments/](experiments/):
- `hospital_base_template.yaml` / `corruption_base_template.yaml` are templates
- `*_private.yaml` files contain actual API keys (not in git)

The [src/config/config.py](src/config/config.py) module loads configs into a `RAGConfig` dataclass.

**Key config parameters**:
- `embedding_model`: Base model or fine-tuned model path
- `top_k_retrieval`: First-stage retrieval count (default 50)
- `top_k_rerank`: Second-stage rerank count (default 10)
- `use_reranker`: Toggle cross-encoder reranking
- `azure_*`: Azure OpenAI credentials and settings

### Two Deployment Patterns

1. **Google Colab** (primary development)
   - Use [all_in_one.ipynb](all_in_one.ipynb) - single notebook with full pipeline
   - Mounts Google Drive for data/index persistence
   - Modes: `"full"` (build index) vs `"quick"` (load cached index)
   - Launches Gradio demo with public URL

2. **HuggingFace Space** (production)
   - Use [gradio_app.py](gradio_app.py) (single model) or [gradio_app_compare.py](gradio_app_compare.py) (model comparison)
   - Indexes uploaded to HF Hub via [upload_to_hf.py](upload_to_hf.py)
   - Space auto-downloads indexes and models on startup

## Common Commands

### 🎯 Recommended: Use Colab Notebooks

**Primary workflow** (no local setup needed):
1. Open [colab_train.ipynb](colab_train.ipynb) in Google Colab
2. Click Runtime → Run all
3. Wait for training and index building (~30-60 mins)
4. Open [colab_evaluation.ipynb](colab_evaluation.ipynb)
5. Run all cells to analyze results (~10-15 mins)
6. Deploy [app.py](app.py) to HuggingFace Space

### Testing the Pipeline (Local)

```bash
# Test imports and configuration
python test_pipeline.py

# Expected output: Tests for imports, config loading, data loading
```

### Installing Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install as package (optional, for imports)
pip install -e .

# Install with development extras
pip install -e .[dev,finetune]
```

### Running Locally

```bash
# 1. Create private config (copy template and add your API keys)
cp experiments/hospital_base_template.yaml experiments/hospital_base_private.yaml
# Edit hospital_base_private.yaml: add azure_api_key, azure_endpoint

# 2. Run test pipeline
python test_pipeline.py

# 3. Launch Gradio demo (from notebooks/launcher.ipynb or gradio_app.py)
python gradio_app.py  # Note: Requires pre-built indexes
```

### Working with Indexes

**Build index programmatically**:
```python
from config.config import load_config
from data.loader import EmailDataLoader
from retrieval.indexer import FAISSIndexBuilder

config = load_config('hospital_base_private')
loader = EmailDataLoader(config)
documents = loader.load_documents()

indexer = FAISSIndexBuilder(config)
indexer.build_index(documents)
indexer.save_index()
```

**Load existing index**:
```python
indexer = FAISSIndexBuilder(config)
index, metadata = indexer.load_index()
```

### Uploading to HuggingFace

```bash
# Upload indexes to HF Hub (requires HF token with write access)
python upload_to_hf.py
# Follow prompts to login and upload
```

## Important Patterns

### Data Format Assumptions

- **Hospital dataset**: First element in thread list is metadata (has `participants` key), rest are emails with `text_latest`
- **Corruption dataset**: All elements are emails with `body_clean` or `body_full`
- Emails are enriched into format: `Subject: ... \nFrom: ... \nTo: ... \nDate: ... \n\n<body>`

### Path Conventions

- Raw data: `data/processed/{dataset}/` (e.g., `data/processed/hospital/threads_with_summary.json`)
- Indexes: `outputs/indexes/{dataset}/` (contains `faiss.index`, `doc_metadata.json`, `config.json`)
- Models: `models/{dataset}/` (for fine-tuned models)
- Configs: `experiments/{dataset}_*.yaml`

### Config Loading

Always use the config name without `.yaml` extension:
```python
config = load_config('hospital_base_private')  # ✅ Correct
config = load_config('hospital_base_private.yaml')  # ❌ Wrong
```

### API Key Management

- Template configs have placeholder values: `YOUR_API_KEY_HERE`
- Private configs (with real keys) use suffix `_private.yaml` and are gitignored
- For Colab: Use Colab Secrets (🔑 sidebar) for `AZURE_API_KEY` and `AZURE_ENDPOINT`
- For HF Space: Add secrets in Space Settings

## Development Workflows

### Colab-First Development

This project is optimized for **Colab-based development**:

1. Edit code locally and push to git
2. In Colab: `!cd /content/RAG-embedding && git pull`
3. Restart kernel and re-run notebook
4. Indexes persist in Google Drive (survive disconnects)

**Key notebook**: [all_in_one.ipynb](all_in_one.ipynb) contains the complete pipeline in one file.

### Adding a New Dataset

1. Place data file in `data/processed/{dataset_name}/`
2. Create config: `experiments/{dataset_name}_base_template.yaml`
3. Update [src/data/loader.py](src/data/loader.py) if format differs
4. Build index using the new config
5. Upload index to HF Hub for deployment

### Fine-Tuning the Embedding Model

1. Train using notebooks in `NewTrain/` or `OldTrain/` (historical reference)
2. Use [train_and_build_index.ipynb](train_and_build_index.ipynb) for the full flow
3. Save fine-tuned model: `model.save_pretrained("models/{dataset}/finetuned")`
4. Update config: Set `use_finetuned: true` and `finetuned_model_path`
5. Rebuild index with fine-tuned embeddings

### Model Comparison

Use [gradio_app_compare.py](gradio_app_compare.py) to compare base vs fine-tuned models:
- Side-by-side results
- Metrics: top score, average score
- See [MODEL_COMPARISON_GUIDE.md](MODEL_COMPARISON_GUIDE.md) for setup

## Troubleshooting

### ModuleNotFoundError in notebooks

The notebooks expect to be run from project root or with proper sys.path setup:
```python
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd()  # Adjust if needed
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
```

### CUDA out of memory

Reduce `batch_size` in config (default 32) or use CPU:
```python
device = "cpu"  # Force CPU if GPU memory insufficient
```

### Index/config hash mismatch

If you change the data or model, rebuild the index completely:
```bash
rm -rf outputs/indexes/{dataset}/*
# Then rebuild
```

### FileNotFoundError for data files

- Verify data path in config matches actual file location
- In Colab: Ensure Google Drive is mounted and data uploaded to correct path
- Check config with: `config.dataset.data_path.exists()`

## File Structure Quick Reference

```
RAG-embedding/
├── 🚀 MAIN WORKFLOWS (Start here!)
│   ├── colab_train.ipynb        # ⭐ One-click training (Run in Colab)
│   ├── colab_evaluation.ipynb   # ⭐ Evaluation & analysis (Run in Colab)
│   └── app.py                   # ⭐ HF Space deployment (model comparison)
│
├── src/                          # Core modules (import from here)
│   ├── config/                   # Configuration management
│   ├── data/                     # Data loading (EmailDataLoader)
│   ├── retrieval/                # FAISS indexing, retrieval, reranking
│   ├── generation/               # Azure OpenAI integration
│   └── app/                      # Gradio UI components
│
├── experiments/                  # YAML configs (per-dataset)
├── data/processed/               # Raw email data (JSON files) - LOCAL ONLY
├── outputs/indexes/              # Built FAISS indexes - LOCAL ONLY
├── models/                       # Fine-tuned models - LOCAL ONLY
│
├── 📓 Legacy notebooks (reference)
│   ├── all_in_one.ipynb         # Old single notebook
│   ├── train_and_build_index.ipynb
│   └── notebooks/               # Step-by-step notebooks
│
├── gradio_app.py                # Old single-model deployment
├── gradio_app_compare.py        # Old comparison deployment
├── upload_to_hf.py              # Manual upload script
├── test_pipeline.py             # Quick sanity check
└── requirements.txt             # Dependencies
```

**Note**: With the new workflow, data/indexes/models are stored in **Google Drive** at:
```
/content/drive/MyDrive/Epiq Project/pipeline/
├── data/processed/
├── faiss_index/
├── models/
└── evaluation_results/
```

## Azure OpenAI Integration

The generation module uses Azure OpenAI (not standard OpenAI API):
- Endpoint format: `https://YOUR-RESOURCE.openai.azure.com/`
- Deployment name (model): typically `gpt-4.1`
- API version: `2024-12-01-preview` (configurable)

**Prompting strategy**: System prompt emphasizes answering only from retrieved email content, no hallucination.

## Notes

### New Workflow (Recommended)
- ✅ Use [colab_train.ipynb](colab_train.ipynb) for complete training pipeline
- ✅ Use [colab_evaluation.ipynb](colab_evaluation.ipynb) for analysis
- ✅ Use [app.py](app.py) for HuggingFace Space deployment
- ✅ All data/models stored in Google Drive: `/content/drive/MyDrive/Epiq Project/pipeline/`
- ✅ HuggingFace username: `ChenyuEcho`
- ✅ Training uses LoRA for efficient fine-tuning

### Legacy Notes
- `all_in_one.ipynb`, `train_and_build_index.ipynb` are older versions (kept for reference)
- `ORIE-5981-RAG/` subdirectory contains older step-by-step notebooks (historical reference)
- Private configs (`*_private.yaml`) are gitignored - always copy from template
- Main branch is `main` (use this for PRs)
- `all_in_one_train.ipynb` in git status is untracked legacy file

### Path Configuration
**Google Drive paths** (used in Colab notebooks):
- Data: `/content/drive/MyDrive/Epiq Project/pipeline/data/processed/`
- Indexes: `/content/drive/MyDrive/Epiq Project/pipeline/faiss_index/`
- Models: `/content/drive/MyDrive/Epiq Project/pipeline/models/`

**HuggingFace repos**:
- Indexes: `ChenyuEcho/rag-indexes`
- Models: `ChenyuEcho/rag-finetuned-hospital`, `ChenyuEcho/rag-finetuned-corruption`
