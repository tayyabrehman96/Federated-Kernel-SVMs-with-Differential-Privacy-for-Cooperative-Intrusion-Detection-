# Version control policy

This repository is intentionally **small and text-centric**: manuscript sources, Python/notebooks, Markdown notes, and **small** numeric artefacts (e.g. `results/metrics.json`, `.txt` thresholds, `.tex` row snippets).

## Tracked

- `template.tex`, `ref.bib`, `response_to_reviewers.md`
- `results/*` as present (JSON / TeX helpers)
- `experiments/**/*.py`, `experiments/**/*.ipynb`, `experiments/requirements.txt`
- `experiments/**/*.md` (reports / READMEs)
- `experiments/edge_iiot/data/.gitkeep`
- Root docs: `README.md`, `DATASETS.md`, `AUTHORS.md`, `CITATION.cff`, `.gitignore`

## Not tracked (local or release only)

- **Benchmark CSVs** under `experiments/CICIDS2017/`, `experiments/CICIDS2023/`, `experiments/edge_iiot/data/`, `experiments/enhanced_synthetic_cyber_attack/*.csv`
- **Figures**: `*.png`, `*.jpg`, … (regenerate from scripts or export from LaTeX build)
- **PDFs** (camera-ready, other submissions)
- **Saved XGBoost boosters** `xgb_cicids2023.json`, `enh_synth_xgb.json` (retrain to reproduce)
- MDPI **`Definitions/`** class bundle (obtain from the publisher)

For bulky reproducibility bundles, use **GitHub Releases** or **Zenodo** and link the DOI or URL from `README.md` / `DATASETS.md`.

## First clone

After cloning, create local folders for data and figures as described in `DATASETS.md` and in `template.tex` (`figures/`, etc.). Nothing in Git replaces downloading licensed datasets.
