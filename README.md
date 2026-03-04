# RAG Email System

Retrieval-Augmented Generation (RAG) system for querying email datasets with semantic search and LLM generation.

---

## 🚀 Quick Start

### **One file. Run in Colab. Done.**

1. Open **[all_in_one.ipynb](all_in_one.ipynb)** in Google Colab
2. Run all cells from top to bottom
3. Wait ~15 mins for first index build
4. Get public Gradio URL to query emails

**That's it!**

---

## 📋 What It Does

- **Retrieval**: FAISS search (top 50) → Cross-Encoder rerank (top 10)
- **Generation**: Azure OpenAI GPT-4 answers from retrieved emails
- **UI**: Gradio web interface with shareable URL
- **Persistence**: Auto-save to Google Drive, reload in 1 second

---

## 📁 Project Structure

```
RAG-embedding/
├── all_in_one.ipynb          # ⭐ USE THIS - everything in one file
├── src/
│   ├── config/               # Config system
│   ├── data/                 # Data loaders
│   ├── retrieval/            # FAISS + reranking
│   ├── generation/           # Azure OpenAI
│   └── app/                  # Gradio UI
└── experiments/              # YAML configs
    ├── hospital_base_template.yaml
    └── corruption_base_template.yaml

Google Drive:
├── Capstone-Data/            # Your data (upload once)
│   ├── hospital/threads_with_summary.json
│   └── corruption/emails_group_by_thread.json
└── Capstone-Outputs/         # Auto-saved indexes
    └── indexes/
```

---

## 🛠️ One-Time Setup

### 1. Upload Data to Google Drive

```
Google Drive/Capstone-Data/
├── hospital/threads_with_summary.json      (28MB, ~9,300 emails)
└── corruption/emails_group_by_thread.json  (3.3MB, ~800 threads)
```

### 2. Add Colab Secrets

In Colab: Click 🔑 **Secrets** (left sidebar) → Add:
- `AZURE_API_KEY`
- `AZURE_ENDPOINT`

---

## 📖 Usage

### Run all_in_one.ipynb

**First time (MODE="full")**:
- Builds index (~15 mins)
- Saves to Drive
- Launches demo

**Next time (MODE="quick")**:
- Loads index (1 sec)
- Launches demo immediately

### Change Settings

In configuration cell:
```python
MODE = "full"          # "full" or "quick"
DATASET = "hospital"   # "hospital" or "corruption"
```

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

## 📊 Performance

| Task | Time | Notes |
|------|------|-------|
| **Build index (hospital)** | ~15 min | T4 GPU, 9,300 docs |
| **Build index (corruption)** | ~2 min | T4 GPU, 800 docs |
| **Load index from Drive** | 1 sec | Subsequent runs |
| **Query (end-to-end)** | 3-5 sec | Retrieve + rerank + generate |

---

## 🎯 Tips

- **Save time**: Use `MODE="quick"` after first run
- **Switch datasets**: Change `DATASET="corruption"` in config cell
- **Indexes persist**: Survive Colab disconnects (saved to Drive)

---

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'data.loader'
- **Fix**: Re-run all cells from top (setup must run first)

### FileNotFoundError: Config file not found
- **Fix**: Use `load_config('hospital_base')` NOT `'experiments/hospital_base.yaml'`

### CUDA out of memory
- **Fix**: Reduce `batch_size` in config

### Colab disconnected
- **No problem!** Re-run with `MODE="quick"` to load saved index

---

## 📝 Tech Stack

- FAISS (vector search)
- Qwen3-Embedding-0.6B (embeddings)
- Cross-Encoder (reranking)
- Azure OpenAI GPT-4.1 (generation)
- Gradio (UI)
- Google Colab T4 GPU
- Google Drive (storage)

---

## 📚 Key Modules

**src/config/config.py**: Config management
**src/data/loader.py**: Multi-format data loading
**src/retrieval/indexer.py**: FAISS index building
**src/retrieval/retriever.py**: Hybrid retrieval pipeline
**src/retrieval/reranker.py**: Cross-Encoder reranking
**src/generation/rag_generator.py**: Azure OpenAI integration
**src/app/gradio_app.py**: Web UI

---

## 📄 License

Academic project - Cornell University

---

**Built for efficient email search and analysis** 🎓
