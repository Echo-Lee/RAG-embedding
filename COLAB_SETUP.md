# Colab Setup Guide

This guide explains how to run the RAG pipeline in Google Colab or with a local Colab connection.

## Setup Methods

### Method 1: Local Colab Connection (Recommended)

Since you've already set up local Colab connection:

1. **Start Colab locally**
   ```bash
   # Your existing local Colab setup
   ```

2. **Open launcher.ipynb in Colab**
   - File → Open notebook
   - Navigate to `notebooks/launcher.ipynb`

3. **Run the notebook**
   - All dependencies will be installed automatically
   - Project files are already accessible via your local filesystem

### Method 2: Google Colab (Cloud)

For sharing with others via cloud Colab:

1. **Upload project to Google Drive**
   ```
   Google Drive/
   └── Capstone-Project/
       ├── src/
       ├── data/
       ├── experiments/
       └── notebooks/
   ```

2. **Mount Drive in Colab**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import os
   os.chdir('/content/drive/MyDrive/Capstone-Project')
   ```

3. **Open launcher.ipynb from Drive**

## Workflow

### First Time Setup (Full Pipeline)

```python
# In launcher.ipynb
MODE = "full"
DATASET = "hospital"  # or "corruption"
```

This will:
1. Install dependencies (~2 minutes)
2. Load embedding model (~1 minute)
3. Encode all documents (~15-20 minutes on T4 GPU)
4. Build FAISS index
5. Save index to `outputs/indexes/{dataset}/`
6. Launch Gradio demo

### Subsequent Runs (Quick Start)

Once index is built:

```python
MODE = "quick"
DATASET = "hospital"
```

This will:
1. Load existing index (~10 seconds)
2. Initialize retriever and generator
3. Launch Gradio demo immediately

## GPU Recommendations

| Task | GPU Needed | Time (T4) | Time (CPU) |
|------|-----------|-----------|------------|
| Index Building | ✅ Yes | ~15 min | ~2 hours |
| Retrieval | ❌ No | <1 sec | <1 sec |
| Reranking | ⚠️ Optional | <1 sec | ~2 sec |
| Generation | ❌ No | ~3 sec | ~3 sec |

**Recommendation**: Use GPU for index building, CPU is fine for inference.

## Colab Tips

### Enable GPU
1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4)
3. Save

### Check GPU Status
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Monitor Memory
```python
# Check RAM
!free -h

# Check GPU memory
!nvidia-smi
```

### Gradio Share Link
When you run the demo, Gradio will generate a public URL:
```
Running on public URL: https://xxxxx.gradio.live
```

Share this link with others to let them try your RAG system!

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```yaml
batch_size: 16  # Default is 32
```

### Slow Encoding
Check if GPU is being used:
```python
# Should print: cuda
print(model.device)
```

### Connection Timeout
For large datasets, increase Gradio timeout:
```python
demo.launch(share=True, server_port=7860, server_name="0.0.0.0",
            show_error=True, quiet=False)
```

### Session Disconnect
Colab sessions disconnect after ~12 hours. To preserve work:
- Save index files to Drive
- Use Quick Start mode to reload

## File Persistence

Files saved during execution:

```
outputs/
└── indexes/
    ├── hospital/
    │   ├── faiss.index         # Vector index (~200MB)
    │   ├── doc_metadata.json   # Document metadata (~50MB)
    │   └── config.json         # Config cache
    └── corruption/
        ├── faiss.index
        ├── doc_metadata.json
        └── config.json
```

These files are automatically saved and can be reused in future sessions.

## Performance Benchmarks

### Hospital Dataset (~9,300 documents)
- Encoding: ~15 min (T4 GPU)
- Index size: ~190 MB
- Search latency: <100ms

### Corruption Dataset (~800 summaries)
- Encoding: ~2 min (T4 GPU)
- Index size: ~18 MB
- Search latency: <50ms

## Next Steps

After successful setup:
1. ✅ Build indexes for both datasets
2. ✅ Test retrieval with example queries
3. ✅ Compare with/without reranker
4. ✅ Share Gradio link for demo
5. 🔄 Fine-tune embeddings (optional)
