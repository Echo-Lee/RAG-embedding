# Quick Git Push Guide

## Before You Push

### ⚠️ Security Check

Make sure API keys are not in tracked configs:

```bash
# Check what will be committed
git status

# Make sure these files show up (they should be gitignored):
# - experiments/hospital_base_private.yaml
# - experiments/corruption_base_private.yaml
# - data/processed/ (large files)
```

If they appear in `git status`, they will be committed! Stop and fix .gitignore first.

---

## Push to GitHub

### First Time Setup

```bash
cd "C:\Users\25765\Desktop\Cornell\Capstone Project"

# Initialize git (if not done)
git init

# Stage all files
git add .

# Check what's staged (should NOT see data files or private configs)
git status

# Commit
git commit -m "Initial commit: RAG pipeline with dual dataset support"

# Add your GitHub repo
git remote add origin https://github.com/Echo-Lee/RAG-embedding.git

# Push
git branch -M main
git push -u origin main
```

### Subsequent Updates

```bash
git add .
git commit -m "Update: description of your changes"
git push
```

---

## What Gets Pushed ✅

```
src/                              # All source code
notebooks/
  ├── launcher.ipynb              # Main launcher
  └── colab_setup.ipynb           # Colab setup helper
experiments/
  ├── hospital_base_template.yaml     # Template (no API key)
  └── corruption_base_template.yaml   # Template (no API key)
requirements.txt
setup.py
README.md
*.md files (documentation)
.gitignore
```

## What Stays Local ❌

```
data/processed/                   # 31MB of data (too large)
experiments/*_private.yaml        # Configs with real API keys
outputs/                          # Generated indexes
models/                           # Trained models
__pycache__/                      # Python cache
.ipynb_checkpoints/              # Jupyter checkpoints
```

---

## After Pushing

### In Colab

1. Open a new Colab notebook
2. Run `notebooks/colab_setup.ipynb` first
3. Then run `notebooks/launcher.ipynb`

Or follow [COLAB_GIT_SETUP.md](COLAB_GIT_SETUP.md) for detailed instructions.

---

## Data Upload to Google Drive

Since data files are too large for Git:

1. **Upload to Google Drive**:
   ```
   Google Drive/
   └── Capstone-Data/
       ├── hospital/
       │   └── threads_with_summary.json        (28MB)
       └── corruption/
           └── emails_group_by_thread.json      (3.3MB)
   ```

2. **In Colab**: Link via `colab_setup.ipynb`

---

## Troubleshooting

### "API keys in git history"

If you accidentally committed API keys:

```bash
# Remove from current commit
git rm --cached experiments/hospital_base.yaml
git rm --cached experiments/corruption_base.yaml
git commit --amend

# Or rewrite history (be careful!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch experiments/*_base.yaml" \
  --prune-empty --tag-name-filter cat -- --all
```

### "Data files too large"

GitHub has 100MB file limit. Our data is small enough (31MB total), but still shouldn't be in git.

Options:
- ✅ Use Google Drive (recommended)
- Use Git LFS (costs money)
- Use external file host

---

## Quick Checklist

Before pushing:

- [ ] Run `git status` - no data files or private configs?
- [ ] Check `.gitignore` - includes `*_private.yaml`?
- [ ] Template configs created - no real API keys?
- [ ] Data uploaded to Drive - ready for Colab?
- [ ] README updated - instructions clear?

After pushing:

- [ ] Clone in Colab works?
- [ ] Data accessible from Drive?
- [ ] API keys configured?
- [ ] Pipeline runs successfully?

---

## Your Workflow

**Local Development**:
```bash
# Make changes
git add .
git commit -m "Your change description"
git push
```

**Use in Colab**:
```python
# Pull latest changes
!git pull origin main

# Or fresh clone
!git clone YOUR_REPO_URL
```

**Share with Others**:
1. Share GitHub repo URL
2. Share Google Drive folder (with read access)
3. Provide Azure API key separately (email/message, NOT in code!)

---

## Summary

| Item | Storage | Access |
|------|---------|--------|
| **Code** | GitHub | `git clone` |
| **Data** | Google Drive | Mount in Colab |
| **API Keys** | Colab Secrets | `userdata.get()` |
| **Outputs** | Generated in Colab | Save to Drive |

This keeps your repo:
- 🔒 Secure (no API keys)
- 📦 Small (no large files)
- 🚀 Fast (quick to clone)
- 🤝 Shareable (others can use it)
