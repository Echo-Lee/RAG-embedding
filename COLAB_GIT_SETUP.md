# Colab Git Setup Guide

## Overview

Since Colab cannot directly access your local files, we use Git to sync the project.

## Strategy

- ✅ **Code**: Push to Git (small files)
- ❌ **Data**: Don't push to Git (too large, use Google Drive or download separately)
- ❌ **API Keys**: Don't push to Git (use template configs instead)

---

## Step 1: Push to GitHub (Local)

### 1.1 Initialize Git (if not already)

```bash
cd "C:\Users\25765\Desktop\Cornell\Capstone Project"
git init
git add .
git commit -m "Initial commit: RAG pipeline implementation"
```

### 1.2 Add Remote and Push

```bash
# Replace with your GitHub repo URL
git remote add origin https://github.com/Echo-Lee/RAG-embedding.git
git branch -M main
git push -u origin main
```

### 1.3 What Gets Pushed

✅ Pushed to Git:
```
src/                  # All code
notebooks/            # Launcher notebook
experiments/          # Template configs (NO API keys)
requirements.txt
README.md
*.md files
```

❌ NOT Pushed (see .gitignore):
```
data/processed/       # Large data files (28MB + 3.3MB)
outputs/              # Generated indexes
experiments/*_private.yaml  # Configs with API keys
models/               # Trained models
```

---

## Step 2: Use in Colab

### 2.1 Clone Repository

```python
# In Colab notebook first cell
!git clone https://github.com/Echo-Lee/RAG-embedding.git
%cd YOUR_REPO
```

### 2.2 Upload Data to Google Drive

**Option A: Manual Upload**
1. Upload your data files to Google Drive:
   ```
   Google Drive/
   └── Capstone-Data/
       ├── hospital/
       │   └── threads_with_summary.json
       └── corruption/
           └── emails_group_by_thread.json
   ```

2. Mount Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # Create symlinks
   !mkdir -p data/processed
   !ln -s /content/drive/MyDrive/Capstone-Data/hospital data/processed/hospital
   !ln -s /content/drive/MyDrive/Capstone-Data/corruption data/processed/corruption
   ```

**Option B: Download from URL** (if you have a file host)
```python
!mkdir -p data/processed/hospital data/processed/corruption
!wget -O data/processed/hospital/threads_with_summary.json "YOUR_FILE_URL"
!wget -O data/processed/corruption/emails_group_by_thread.json "YOUR_FILE_URL"
```

### 2.3 Setup API Keys

Create config with your API key:

```python
# Copy template and add your key
!cp experiments/hospital_base_template.yaml experiments/hospital_base.yaml

# Edit the file to add your API key
from google.colab import files
import yaml

# Or set directly in Python
config = yaml.safe_load(open('experiments/hospital_base_template.yaml'))
config['azure_api_key'] = 'YOUR_ACTUAL_KEY'
config['azure_endpoint'] = 'YOUR_ACTUAL_ENDPOINT'

with open('experiments/hospital_base.yaml', 'w') as f:
    yaml.dump(config, f)
```

---

## Step 3: Complete Colab Setup

Now you can run the launcher:

```python
# Install dependencies
!pip install -q sentence-transformers faiss-cpu gradio pyyaml openai tqdm

# Add to path
import sys
sys.path.insert(0, '/content/YOUR_REPO/src')

# Run pipeline
MODE = "full"
DATASET = "hospital"

# ... rest of launcher code
```

---

## Alternative: Use Colab Secrets (Recommended)

Store API keys in Colab Secrets instead of files:

```python
from google.colab import userdata

# Store once: Secrets → Add new secret → name: "AZURE_API_KEY"
api_key = userdata.get('AZURE_API_KEY')
api_endpoint = userdata.get('AZURE_ENDPOINT')

# Load config and override
import yaml
config_dict = yaml.safe_load(open('experiments/hospital_base_template.yaml'))
config_dict['azure_api_key'] = api_key
config_dict['azure_endpoint'] = api_endpoint

# Save to temporary config
with open('experiments/hospital_base.yaml', 'w') as f:
    yaml.dump(config_dict, f)
```

---

## Data Size Considerations

Your data files are large for Git:
- Hospital: 28MB
- Corruption: 3.3MB

**Recommended approach**: Use Google Drive for data

**Alternative with Git LFS** (if you want to version data):
```bash
# Local machine
git lfs install
git lfs track "data/processed/**/*.json"
git add .gitattributes
git add data/
git commit -m "Add data with LFS"
git push
```

Then in Colab:
```bash
!git lfs pull
```

---

## Complete Colab Notebook Template

Create this as `notebooks/colab_setup.ipynb`:

```python
# Cell 1: Clone repo
!git clone https://github.com/Echo-Lee/RAG-embedding.git
%cd YOUR_REPO

# Cell 2: Mount Drive and link data
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p data/processed
!ln -s /content/drive/MyDrive/Capstone-Data/hospital data/processed/hospital
!ln -s /content/drive/MyDrive/Capstone-Data/corruption data/processed/corruption

# Verify data
!ls -lh data/processed/*/

# Cell 3: Install dependencies
!pip install -q sentence-transformers faiss-cpu gradio pyyaml openai tqdm

# Cell 4: Setup API keys (using Colab Secrets)
from google.colab import userdata
import yaml

api_key = userdata.get('AZURE_API_KEY')
endpoint = userdata.get('AZURE_ENDPOINT')

for dataset in ['hospital', 'corruption']:
    config = yaml.safe_load(open(f'experiments/{dataset}_base_template.yaml'))
    config['azure_api_key'] = api_key
    config['azure_endpoint'] = endpoint

    with open(f'experiments/{dataset}_base.yaml', 'w') as f:
        yaml.dump(config, f)

print("✅ Setup complete!")

# Cell 5: Now open launcher.ipynb and run
print("Open notebooks/launcher.ipynb and run cells")
```

---

## Updating Code from Local

When you make changes locally:

```bash
# Local machine
git add .
git commit -m "Update code"
git push

# In Colab
!git pull origin main
```

---

## Summary

| Component | Storage | Access in Colab |
|-----------|---------|----------------|
| Code | GitHub | `git clone` |
| Data | Google Drive | Mount + symlink |
| API Keys | Colab Secrets | `userdata.get()` |
| Outputs | Colab (temporary) | Save to Drive before session ends |

This approach keeps your repo clean, secure, and easy to share!
