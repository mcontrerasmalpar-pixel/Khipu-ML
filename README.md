# Khipu-ml

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24619584.svg)](https://doi.org/10.5281/zenodo.24619584)

Machine learning on Inka khipus — knotted-cord records of the Inka state that remain undeciphered.

[![arXiv](https://img.shields.io/badge/arXiv-2607.00185-b31b1b.svg)](https://arxiv.org/abs/2607.00185)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-notebook-20BEFF.svg)](https://www.kaggle.com/code/macmaky/khipu-ml)

Maria Contreras ([Disentangle Lab](https://github.com/Disentangle-Lab)) · paper [arXiv:2607.00185](https://arxiv.org/abs/2607.00185) · data [Open Khipu Repository](https://doi.org/10.5281/zenodo.18025748)

The pipeline turns the public OKR (619 khipus, 54,403 cords, 110,677 knots) into 27 structural features per specimen, then clusters, classifies provenance, and replicates a published Santa Valley finding — without touching the objects.

| | |
|---|---|
| Corpus | **619** khipus · **27** features |
| Clusters | **3** groups, silhouette **0.769** |
| Imperial style | XGBoost F1 **0.86** (Inka Late Horizon) |
| Santa Valley | Recto/verso moiety **reproduced** from OKR alone |

## What it finds

**Imperial khipus are a tight cluster.** 17 Late Horizon specimens sit apart from the rest of the corpus. SHAP ranks cord twist, short uniform length, high knot density, and a restricted color palette as the signature of centralized manufacture. S-twist dominates imperial cords (85.3%).

**One cluster is colonial collecting, not a region.** Cluster 2 (160 khipus) is dominated by 19th-century European and North American museums. Twist-ratio vs the Central Coast majority differs at *p* ≈ 3.7×10⁻²⁷⁶, much of it unrecorded twist coded `U`. The OKR is not a neutral sample of Inka khipus.

**Santa Valley moiety structure holds in the public database.** Medrano and Urton (2018) argued that attachment direction (recto/verso) encodes hanan/hurin. From OKR `ATTACHMENT_TYPE` only, five of six Santa Valley khipus (KH0323–KH0328) are pure R or pure V; **KH0326 is the only mixed specimen** — the same khipu the original study flagged. Aggregate R/V is 49%/51% vs their ~47%/53%. This is the structural claim, not the 367-peso tribute total.

| OKR | Recto | Verso |
|---|---:|---:|
| KH0323 | 310 | 0 |
| KH0324 | 0 | 54 |
| KH0325 | 0 | 207 |
| KH0326 | 40 | 69 |
| KH0327 | 0 | 93 |
| KH0328 | 56 | 0 |

Coastal classes confuse each other (weighted F1 0.46 on 135 labeled specimens). Knot-type n-grams add nothing (ΔF1 ≈ −0.006).

```mermaid
flowchart LR
  OKR["OKR SQLite"] --> Feat["27 features"]
  Feat --> Clust["UMAP + HDBSCAN"]
  Feat --> Clf["XGBoost + SHAP"]
  Feat --> SV["Santa Valley R/V"]
  Clust --> Out["3 clusters"]
  Clf --> Imp["imperial F1 0.86"]
```

## Run it

```bash
git clone https://github.com/Disentangle-Lab/Khipu-ML.git
cd khipu-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download [`khipu.db`](https://doi.org/10.5281/zenodo.18025748) into `data/`, then open `notebooks/01_pipeline_khipu.ipynb` (v1, same notebook on [Kaggle](https://www.kaggle.com/code/macmaky/khipu-ml)) or `notebooks/02_validations_v2.ipynb` (v2 audit & B7).

## v2 Audit & Section B7 (`jclindaniel` audit)

[`notebooks/02_validations_v2.ipynb`](notebooks/02_validations_v2.ipynb) audits v1 against documentation and cataloger confounds on OKR Zenodo release **v2.1.0** ([DOI 10.5281/zenodo.18025748](https://doi.org/10.5281/zenodo.18025748)) and exports versioned summary tables to `results/`.

Sequential filter order in B7 (saved in [`results/excluded_khipus_okr_2.1.0.csv`](results/excluded_khipus_okr_2.1.0.csv)):
1. `empty_no_cords` (`n = 7`, `KH0601–KH0607`, zero cords in `cord`) → `612`
2. `pct_twist_U_ge_0.20` (`n = 145`, $\ge 20\%$ unrecorded cord twist) → `467`
3. `created_by_jclindaniel` (`n = 37`, treated as non-comparable twist without recoding; S↔Z cause pending confirmation from OKR / Clindaniel) → `430`

To reproduce B7 end-to-end:
```bash
pip install -r requirements.txt
export KHIPU_DB_PATH=data/khipu.db   # optional; auto-detects data/ or downloads Zenodo 2.1.0
# Run notebooks/02_validations_v2.ipynb (set B7_RUN_NULLS = True in the B7 config cell to recompute all 1,000 + 1,000 null simulations in both UMAP and feature space via src/b7_nulls.py)
```

Key versioned outputs in `results/`:
- `excluded_khipus_okr_2.1.0.csv` — reproducible exclusion list (`n = 189` excluded, `430` clean)
- `table_b2_1000.csv` — `N_SIM = 1,000 + 1,000` permutation & Gaussian null models in **both** UMAP 2D and original feature space (`sil_FEAT`)
- `table_b7_jclindaniel_sites.csv`, `table_b7_progression.csv`, `table_b7_doc_tree.csv`, `table_b7_threshold_sweep.csv`, `table_hyperparam_sweep.csv`
- `table_sensitivity_matrix.csv`, `table_twist_by_cluster.csv` (cord-level `S/Z/U/blank` correcting v1's 85.3%), `table_missing_aware_comparison.csv`, `table_santa_valley_rv.csv`

**Limitations:** No positive control is included yet; `"Inka, Late Horizon"` (`n = 18` in v1, `n = 3` in `n = 430`) comes from a single museum (Dallas Museum of Art) and cannot be evaluated independently in `n = 430`; the cause of the observed S/Z twist contrast by `CREATED_BY` in Incahuasi and Armatambo remains an open hypothesis pending confirmation from the OKR team / Jon Clindaniel.

```
data/          khipu.db (gitignored)
notebooks/     01_pipeline_khipu.ipynb (v1), 02_validations_v2.ipynb (v2 + B7)
results/       versioned small CSV tables & exclusion list (OKR 2.1.0)
src/           loader, features, utils, b7_nulls
outputs/       figures & full JSON (gitignored)
```

## Cite

```bibtex
@misc{contreras2026khipu,
  title={Structural Pattern Mining in Inka Khipus: Unsupervised Clustering, Provenance Classification, and a Computational Validation of the Santa Valley Match},
  author={Contreras, Maria},
  year={2026},
  eprint={2607.00185},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2607.00185}
}
```

```bibtex
@dataset{open_khipu_repository,
  title = {Open Khipu Repository},
  doi   = {10.5281/zenodo.18025748},
  url   = {https://doi.org/10.5281/zenodo.18025748}
}
```
