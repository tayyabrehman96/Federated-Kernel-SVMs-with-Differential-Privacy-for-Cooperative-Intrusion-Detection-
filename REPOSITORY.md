# GitHub distribution policy (what to push)

Official repository: **[github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-](https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-)**

This document fixes **what belongs on Git** for developers and what should stay local or on a release / Zenodo bundle. It matches the code and LaTeX currently in this project—not every file on your disk should be pushed.

## Training / experiment modes in this codebase

| Mode | Script(s) | Data | What it reproduces |
|------|-----------|------|-------------------|
| **Federated synthetic (proxy FL tables)** | `experiments/run_federated_revision_tables.py` → `generate_revision_results.py` | None (synthetic 84-D binary FL) | Revision LaTeX helpers, `results/metrics.json`, non-IID / DP / Byzantine style plots under `figures/` |
| **Centralised XGBoost on flow CSVs** | `experiments/run_cic_flow_xgboost_baseline.py`, `cicids2023_xgboost_trainer.py` | PCAP-flow CSV folder (`CIC_FLOW_BENCHMARK_DIR`) | IoT/IDS-flow baseline; artefacts in `CICIDS2023_code/` |
| **CICIDS2017 notebook** | `experiments/CICIDS2017_code/train_cicids2017.ipynb` | `experiments/CICIDS2017/*.csv` | Per-paper 2017 experiments (as implemented in the notebook) |
| **Enhanced synthetic** | `experiments/enhanced_synthetic_cyber_attack/train_and_visualize_enhanced_synthetic.py` | `Enhanced_Synthetic_Cyber_Attack_Dataset.csv` | Auxiliary synthetic benchmark |
| **Edge-IIoT** | *Add when same pipeline as manuscript is committed* | `experiments/edge_iiot/data/` | Paper table/figure numbers for Edge-IIoT |

The **full MDPI manuscript** (all centralised + federated runs on real CICIoT2023 and Edge-IIoT) may use additional scripts or notebooks not yet in this folder; the table above is what **this repository currently ships**.

## Include in Git (recommended)

- **Documentation:** `README.md`, `DATASETS.md`, `REPOSITORY.md`, `AUTHORS.md`, `CITATION.cff`, `.gitignore`
- **Paper sources (if you choose to publish them):** `template.tex`, `ref.bib`, `response_to_reviewers.md`, `results/*.tex`, `results/metrics.json`
- **All Python:** `experiments/**/*.py`, utility scripts at repo root if any
- **Notebooks:** `experiments/**/*.ipynb`
- **Dependencies:** `experiments/requirements.txt` (extend if XGBoost etc. are required for your README quickstarts)
- **Small run artefacts** (optional but useful): `experiments/**/*.{json,txt}` for metrics/thresholds, `experiments/**/*.md` reports, pre-trained small JSON models like `xgb_cicids2023.json` if license/size allow
- **Layout stubs:** `experiments/edge_iiot/data/.gitkeep`, any `README.md` under `experiments/*/`

## Exclude from Git (default; keep local or use Releases / Zenodo)

- **Raw benchmark CSVs** under `experiments/CICIDS2017/`, `experiments/CICIDS2023/`, and large `*.csv` in `enhanced_synthetic_cyber_attack/` if they push the repo over **~50–100 MB** or violate redistribution rules
- **MDPI class files:** `Definitions/` (copyrighted template—developers obtain from MDPI)
- **Generated LaTeX noise:** `*.aux`, `*.log`, `*.pdf` builds
- **Secrets:** API keys, machine-specific paths (use `CIC_FLOW_BENCHMARK_DIR`, not hard-coded `D:\...` in committed code)
- **Virtual environments:** `.venv/`, `venv/`

### If you must version large files

Use **[Git LFS](https://git-lfs.github.com/)** or a **[GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)** / **Zenodo** archive, then link the URL in `DATASETS.md`.

## First-time push checklist

1. Confirm `git status` shows no multi-GB CSVs unless intentional.
2. Add remote: `git remote add origin https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-.git`
3. Default branch `main`, `.gitignore` active before first commit.
4. Pin dataset URLs for readers (see `DATASETS.md`); do not rely on your local paths.

## Citation

See `CITATION.cff` and `AUTHORS.md`. After the MDPI article has a DOI, add it to `README.md` and optionally a `preferred-citation` block in `CITATION.cff`.
