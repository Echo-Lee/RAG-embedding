# RAG Email Assistant

A complete Retrieval-Augmented Generation (RAG) system for email search and question answering, with support for custom embedding fine-tuning.

## Features

- **Dense Retrieval**: FAISS-based vector search with Qwen3 embeddings
- **Reranking**: Cross-Encoder reranking for improved accuracy
- **RAG Generation**: Azure OpenAI GPT-4 for answer generation
- **Fine-tuning**: Support for custom embedding model training
- **Interactive UI**: Gradio web interface
- **Multi-dataset**: Supports hospital and corruption email datasets

## Quick Start

### 1. Prepare Data

Data files are already in place (both preprocessed):

```
data/
└── processed/
    ├── hospital/threads_with_summary.json           # Hospital emails (preprocessed)
    └── corruption/emails_group_by_thread.json       # Corruption emails (preprocessed)
```

### 2. Configure (Optional)

Edit experiment config files in `experiments/` if needed:
- `hospital_base.yaml`
- `corruption_base.yaml`

Azure OpenAI credentials are already configured.

### 3. Launch in Colab or Local Runtime

Open `notebooks/launcher.ipynb` and:
- Dependencies will be auto-installed in the first cell
- Select your mode and dataset:
  - **Quick Start**: Load existing index (fast)
  - **Full Pipeline**: Build index from scratch (first time)
  - **Fine-tune**: Train custom embedding model
- Run all cells

### 4. Local Development (Optional)

If developing locally, install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── data/                   # Datasets
│   ├── raw/               # Original data files
│   ├── processed/         # Processed data
│   └── fine_tune/         # Fine-tuning data
├── src/                   # Source code
│   ├── config/            # Configuration management
│   ├── data/              # Data loading
│   ├── models/            # Model training
│   ├── retrieval/         # Retrieval & reranking
│   ├── generation/        # Answer generation
│   └── app/               # Gradio UI
├── notebooks/             # Jupyter notebooks
│   └── launcher.ipynb     # Main entry point
├── experiments/           # Experiment configs (YAML)
├── models/                # Trained models
└── outputs/               # Outputs (indexes, results)
```

## Pipeline Overview

```
User Query
    ↓
[Embedding Model]
    ↓
[FAISS Dense Retrieval] → top-50 candidates
    ↓
[Cross-Encoder Reranker] → top-10 documents
    ↓
[Azure OpenAI GPT-4] → Generated Answer
    ↓
Display to User
```

## Datasets

### Hospital Dataset
- **Location**: `data/processed/hospital/threads_with_summary.json`
- **Format**: Thread-based with full email content
- **Structure**: Thread metadata + individual emails
- **Status**: Preprocessed, ready for indexing
- **Size**: ~9,300 documents

### Corruption Dataset
- **Location**: `data/processed/corruption/emails_group_by_thread.json`
- **Format**: Thread-grouped with full email content
- **Structure**: Thread ID → list of emails
- **Status**: Preprocessed, ready for indexing
- **Size**: ~800 threads with multiple emails each

## Configuration

Key parameters in YAML config files:

```yaml
# Retrieval
top_k_retrieval: 50        # First stage candidates
top_k_rerank: 10           # Final results after reranking

# Models
embedding_model: Qwen/Qwen3-Embedding-0.6B
reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

# Generation
azure_endpoint: your-endpoint
azure_api_key: your-key
```

## Usage Examples

### Python API

```python
from config.config import load_config
from retrieval.retriever import HybridRetriever
from generation.rag_generator import RAGGenerator

# Load config
config = load_config('hospital_base')

# Initialize components
retriever = HybridRetriever(config)
generator = RAGGenerator(config)

# Search and generate answer
docs = retriever.retrieve("What is the main topic?", top_k=5)
answer = generator.generate("What is the main topic?", docs)

print(answer)
```

### Gradio UI

Launch the interactive demo:

```python
from app.gradio_app import create_demo

demo = create_demo(retriever, reranker, generator, config)
demo.launch(share=True)
```

## Colab Usage

When using Google Colab:

1. Upload `src/` folder to Colab or connect via local runtime
2. Open `notebooks/launcher.ipynb`
3. Run cells to set up environment
4. Choose mode and run pipeline

## Advanced Features

### Custom Embeddings

Train fine-tuned embedding models (coming soon):

```python
MODE = "finetune"
DATASET = "hospital"
# Run fine-tuning cells
```

### Reranker Comparison

Test with/without reranker:

```python
# With reranker (better accuracy)
docs = retriever.retrieve(query, use_rerank=True)

# Without reranker (faster)
docs = retriever.retrieve(query, use_rerank=False)
```

## Performance

Expected Hit@K on test sets:

| Model | Hit@1 | Hit@5 | Hit@10 |
|-------|-------|-------|--------|
| Base  | ~65%  | ~85%  | ~90%   |
| Fine-tuned | ~75% | ~92% | ~95% |

## Troubleshooting

### Index not found
Run **Full Pipeline** mode first to build the index.

### CUDA out of memory
Reduce `batch_size` in config file.

### API errors
Check Azure OpenAI credentials in config.

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
