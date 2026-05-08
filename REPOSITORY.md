# Version control policy

The repository is **code-first**: Python, notebooks, Markdown, two reference figures (`pm.png`, `Methodology_SM.jpg`), and **`results/metrics.json`**.

## Tracked

- `README.md`, `DATASETS.md`, `REPOSITORY.md`, `AUTHORS.md`, `CITATION.cff`, `.gitignore`
- `pm.png`, `Methodology_SM.jpg`
- `results/metrics.json` only (no LaTeX table fragments)
- `experiments/**/*.py`, `experiments/**/*.ipynb`, `experiments/requirements.txt`, `experiments/**/*.md`, `experiments/edge_iiot/data/.gitkeep`

## Not tracked (local, Zenodo, or release)

- **Manuscript:** `*.tex`, `ref.bib`, reviewer letters — keep privately or on the journal system
- **Raw / derived CSV** under dataset folders
- **Raster figures** except the two whitelisted images; **PDF** camera-ready
- **Saved XGBoost / other checkpoints** (`xgb_cicids2023.json`, `enh_synth_xgb.json`, etc.)
- **Preprocessing dumps** — preprocessing runs **inside** trainers/notebooks; do not commit intermediate parquet/pickle trees unless via Zenodo with a DOI

## Zenodo & model hosting

Freeze optional bundles (split hashes, configs, permitted checkpoints) on **Zenodo** and add the DOI to `README.md`. Optional **Hugging Face** links for distilled models can be added the same way.
