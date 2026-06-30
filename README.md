# Khipu ML — Structural Pattern Mining in Inka Khipus

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This repository contains an academic research pipeline that applies unsupervised and supervised machine learning to the [Open Khipu Repository (OKR)](https://doi.org/10.5281/zenodo.18025748) — a curated, structured database of Inka khipus (knotted-cord recording devices). The pipeline leverages **UMAP + HDBSCAN** for dimensionality reduction and density-based clustering of cord-level features, and **XGBoost** with SHAP interpretability for classification tasks. The goal is to identify structural and numerical patterns that may inform the ongoing decipherment effort.

## Dataset (OKR)

The Open Khipu Repository is a community-maintained, versioned database of khipu transcriptions in SQLite format. It encodes cord hierarchy, knot types, knot values, pendant colors, and attachment styles for hundreds of museum specimens worldwide.

- **Format:** SQLite (`data/khipu.db`, gitignored — download separately)
- **DOI:** [10.5281/zenodo.18025748](https://doi.org/10.5281/zenodo.18025748)

## Project Structure

```
khipu-ml/
├── data/               # khipu.db goes here (gitignored)
├── notebooks/
│   └── 01_pipeline_khipu.ipynb     # Full pipeline: EDA, features, UMAP+HDBSCAN, XGBoost, SHAP, tuning, Santa Valley validation
├── src/
│   ├── __init__.py
│   ├── loader.py       # SQLite connection and data loading
│   ├── features.py     # Feature engineering utilities
│   └── utils.py        # General-purpose helpers
├── outputs/            # Generated figures and CSVs (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## Pipeline

**Notebook:** `01_pipeline_khipu.ipynb` ([also on Kaggle](https://www.kaggle.com/code/macmaky/khipu-ml))

| Step | Description |
|------|-------------|
| EDA | Schema exploration, missing value analysis, distributions of knot values, cord colors, and attachment metadata |
| Features | Construction of cord-level and khipu-level feature matrices; encoding of categorical variables |
| UMAP + HDBSCAN | UMAP projection to 2D/3D; HDBSCAN clustering → **3 clusters** (silhouette = 0.769); cluster profiling and visualization |
| XGBoost | XGBoost classifier for khipu typology; class imbalance handling via `imbalanced-learn`; cross-validation → **F1 = 0.80** (Inka imperial style) |
| SHAP | SHAP values for global and local feature importance; cluster interpretation; hypothesis generation |
| XGBoost Tuning | `RandomizedSearchCV` hyperparameter search with class weighting → **F1 = 0.86** (tuned, Inka Late Horizon) |
| Confusion Analysis | Confusion matrix and t-tests on Central Coast vs South Coast misclassification |
| Sequence Modeling | TF-IDF over knot-type n-grams (bigrams/trigrams) tested as additional features — no measurable improvement (ΔF1 ≈ −0.006) |
| Santa Valley Validation | Independent computational replication of Medrano & Urton (2018) recto/verso moiety findings using only the public OKR database |

## Results

### Clustering (UMAP + HDBSCAN)

- 3 structurally distinct clusters, silhouette = 0.769
- Cluster 0 (17 khipus): Inka Late Horizon imperial style
- Cluster 1 (442 khipus): Central Coast, Peru — majority style
- Cluster 2 (160 khipus): 19th century European collections

### Classification (XGBoost)

- F1 weighted = 0.46 baseline (7 classes, 135 labeled samples)
- F1 = 0.80 for Inka Late Horizon style
- F1 = 0.86 for Inka Late Horizon style after hyperparameter tuning (`RandomizedSearchCV`, class weighting)
- Knot-type n-gram sequence features tested via TF-IDF: no measurable gain over aggregate features (ΔF1 ≈ −0.006)

### Interpretability (SHAP)

Top features for Inka imperial detection:

1. `knot_dir_Z` — standardized Z-direction knotting
2. `mean_length` — shorter, more uniform cords
3. `n_knots` — higher knot density
4. `std_length` — low variability (high standardization)
5. `color_entropy` — restricted color palette

### Santa Valley Validation (Medrano & Urton, 2018)

Independent computational replication of the Santa Valley khipu group, using only the public OKR database (no physical object access):

- 6 khipus identified in OKR: KH0323–KH0328 (Museo Temple Radicati, Lima)
- Recto/Verso replication: this study (cord-level) R=49.0%, V=51.0% vs. Medrano & Urton (group-level) R=47%, V=53% — consistent
- Moiety structure confirmed: 5 of 6 khipus are purely R or purely V; KH0326 is the sole mixed khipu, matching the original finding exactly

### Key Finding

Inka imperial khipus are ML-detectable with F1=0.86 (tuned).
Construction conventions encode regional identity.
Colonial collection patterns are visible in corpus structure.
The Santa Valley recto/verso moiety pattern reported in the literature is independently reproducible from the public OKR database alone.

### Versioned Results

- **v5:** Cluster 2 deep analysis — twist S identified as Inka imperial manufacturing signature (p=3.7e-276)
- Colonial collection bias confirmed: double layer (geographic + methodological recording bias)
- XGBoost tuning raised F1 for Inka Late Horizon style from 0.80 to 0.86
- Knot-sequence n-gram features evaluated and found not to improve classification
- First independent computational verification of the Santa Valley recto/verso match (Medrano & Urton 2018) from OKR data

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_pipeline_khipu.ipynb` | Full pipeline: EDA, feature engineering, UMAP+HDBSCAN clustering, XGBoost classification + tuning, SHAP interpretability, sequence modeling, Santa Valley validation |

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/khipu-ml.git
cd khipu-ml

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the OKR database
#    Place khipu.db inside the data/ directory.
#    Download from: https://doi.org/10.5281/zenodo.18025748

# 5. Launch Jupyter
jupyter notebook
```

## Citation

If you use this code or the OKR data in your research, please cite the Open Khipu Repository:

```bibtex
@dataset{open_khipu_repository,
  title  = {Open Khipu Repository},
  doi    = {10.5281/zenodo.18025748},
  url    = {https://doi.org/10.5281/zenodo.18025748}
}
```
