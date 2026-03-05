# Archive Directory

This directory contains legacy files that have been replaced by the new streamlined workflow.

## Archived on: 2026-03-05

---

## What's in here:

### `old_notebooks/`
Old training and pipeline notebooks (replaced by `colab_train.ipynb`):
- `all_in_one.ipynb` - Old single notebook with full pipeline
- `all_in_one_train.ipynb` - Old training notebook
- `train_and_build_index.ipynb` - Old training + index building
- `pipeline.ipynb` - Old pipeline notebook
- `RAG.ipynb` - Very old RAG notebook
- `notebooks/` - Old launcher notebooks

### `old_deployments/`
Old deployment files (replaced by `app.py`):
- `gradio_app.py` - Old single-model deployment
- `gradio_app_compare.py` - Old comparison deployment
- `upload_to_hf.py` - Manual upload script (now integrated into `colab_train.ipynb`)

### `old_guides/`
Old documentation (replaced by updated `README.md` and `CLAUDE.md`):
- `DEPLOY_GUIDE.md` - Old Chinese deployment guide
- `MODEL_COMPARISON_GUIDE.md` - Old Chinese comparison guide

### `old_training_data/`
Legacy training experiments:
- `OldTrain/` - Old training data and experiments
- `NewTrain/` - Intermediate training data

---

## Still in main directory (historical reference):

### `ORIE-5981-RAG/`
**Location**: Still in project root (couldn't be moved due to Git submodule)

Original step-by-step notebooks for reference:
- `Step0_data_analysis.ipynb`
- `Step1_email_preprocess.ipynb`
- `Step2_summary_generation.ipynb`
- `Step3_question_generation.ipynb`
- `Step4_fine_tune.ipynb`

---

## New workflow (current):

Use these instead:
1. **Training**: `colab_train.ipynb` - One-click training pipeline
2. **Evaluation**: `colab_evaluation.ipynb` - Analysis and visualization
3. **Deployment**: `app.py` - HuggingFace Space with model comparison

See `CLAUDE.md` for complete documentation.

---

**Note**: These archived files are kept for historical reference only. They are not maintained and may not work with the current codebase.
