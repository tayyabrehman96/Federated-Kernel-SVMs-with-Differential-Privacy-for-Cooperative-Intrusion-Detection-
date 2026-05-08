# Federated kernel SVMs with differential privacy for cooperative IDS

Research artifacts for **Federated Kernel SVMs with Differential Privacy for Cooperative Intrusion Detection in Smart Meter Networks** (MDPI *Sensors*).  

**Code:** [github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-](https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-)

---

### What this repository is

A **source-only** snapshot: LaTeX and BibTeX, Python utilities, one notebook, and small text/JSON table outputs. It does **not** store raw benchmark CSVs, PDFs, or raster figures—those stay on your machine after you download public datasets or rebuild plots.

### Layout

| Path | Purpose |
|------|---------|
| `template.tex`, `ref.bib` | Manuscript sources (you still need the MDPI `Definitions/` class files locally to compile). |
| `response_to_reviewers.md` | Revision correspondence. |
| `results/` | Compact metrics (`metrics.json`) and generated LaTeX row fragments from the federated simulation script. |
| `experiments/` | Federated table generator, XGBoost flow-CSV baseline, dataset path helpers, optional enhanced-synthetic script. |

Author list, emails, and GitHub citation metadata: [`AUTHORS.md`](AUTHORS.md), [`CITATION.cff`](CITATION.cff).

### Requirements

Python 3.10+ recommended. Core stack: `numpy`, `scikit-learn`, `matplotlib`; optional `torch` (see `experiments/requirements.txt`). XGBoost is only needed for the PCAP-flow baseline.

### Run

**Federated simulation (revision tables, no CIC downloads):**

```bash
cd experiments
pip install -r requirements.txt
python run_federated_revision_tables.py
```

**XGBoost baseline on a folder of flow CSVs** (set `CIC_FLOW_BENCHMARK_DIR` if data are not in `experiments/CICIDS2023/`):

```bash
cd experiments
pip install xgboost pandas scikit-learn matplotlib
python run_cic_flow_xgboost_baseline.py
```

**CICIDS2017 notebook:** `experiments/CICIDS2017_code/train_cicids2017.ipynb` (place day CSVs under `experiments/CICIDS2017/` locally; not in Git).

### Datasets (download separately)

Instructions and official links: [`DATASETS.md`](DATASETS.md).  
Benchmarks used in the paper: **CICIDS2017**, **CICIoT2023**, **Edge-IIoTset**.

### Contributing policy

[`REPOSITORY.md`](REPOSITORY.md) describes what belongs in version control versus local-only or Zenodo/Git LFS.

### Suggested GitHub “About” description (one line)

`Federated kernel SVMs + DP for cooperative IDS; MDPI Sensors research code (source only, no bundled datasets).`
