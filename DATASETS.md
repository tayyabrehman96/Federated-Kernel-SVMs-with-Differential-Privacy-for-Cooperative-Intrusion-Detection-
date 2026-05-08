# Datasets — MDPI three-benchmark layout

Use this file when cloning the repo so **directory names match what developers expect**, while staying aligned with `template.tex` (**CICIDS2017**, **CICIoT2023**, **Edge-IIoT**).

## Official download links (readers obtain data here)

| Benchmark | Primary source | Notes |
|-----------|----------------|--------|
| **CICIDS2017** | [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) | Canadian Institute for Cybersecurity (UNB). Follow their current license/terms. |
| **CICIoT2023** | [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | Large-scale IoT attacks; paper text uses this name. Preprocessed flow CSVs are typical for IDS baselines. |
| **Edge-IIoTset** | [IEEE DataPort — Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) | Companion to Ferrag *et al.*, *IEEE Trans. Ind. Informatics* — DOI [10.1109/TII.2022.3155656](https://doi.org/10.1109/TII.2022.3155656). |

Community mirrors (e.g. Kaggle) exist; for citations and reproducibility, prefer the **official** pages above when possible.

## 1. CICIDS2017

- **Expected path:** `experiments/CICIDS2017/`
- **Files:** `*-WorkingHours*.pcap_ISCX.csv` (one CSV per weekday / scenario).
- **Training artefact:** `experiments/CICIDS2017_code/train_cicids2017.ipynb`

## 2. CIC-IoT (paper: CICIoT2023)

The LaTeX source names this benchmark **CICIoT2023**. This repository’s **default** flow-CSV tree for the bundled XGBoost baseline uses a folder named `CICIDS2023` (PCAP-flow naming such as `BenignTraffic*.pcap.csv`). That layout is a **convenience path**, not a substitute for reading the official CICIoT2023 documentation.

- **Default path in code:** `experiments/CICIDS2023/` (or set `CIC_FLOW_BENCHMARK_DIR`)
- **Baseline trainer:** `experiments/CICIDS2023_code/cicids2023_xgboost_trainer.py`

### Using a different extract (e.g. updated 2024 mirror)

Point the trainer at your CSV root:

```bash
cd experiments/CICIDS2023_code
set CIC_FLOW_BENCHMARK_DIR=D:\path\to\your-flow-csv-folder
python cicids2023_xgboost_trainer.py
```

The loader expects one CSV per file under that directory; benign files must be named so that `normalize_label_from_filename` maps them to **Benign** (`BenignTraffic*` in the stock script). If your official CICIoT2023 layout differs, add a small adapter next to the trainer.

## 3. Edge-IIoT

- **Expected path:** `experiments/edge_iiot/data/`
- **Paper reference:** `template.tex` / `ref.bib` (`edgeiiot2022`). Preprocess to the **61-feature** flow schema described in the manuscript before training.
- **This repo:** `data/` is kept in Git with `.gitkeep` only; **do not** commit multi-gigabyte dumps unless you use Git LFS and comply with the dataset license (see [REPOSITORY.md](REPOSITORY.md)).

## Synthetic / auxiliary

- **Enhanced synthetic:** `experiments/enhanced_synthetic_cyber_attack/`
- **Federated synthetic LaTeX helpers:** `experiments/generate_revision_results.py` — entry point `run_federated_revision_tables.py`

## Licensing

Do not redistribute benchmark data on GitHub unless the dataset license allows it. Default practice: **document URLs + folder layout**; keep large CSVs local or in Zenodo/GitHub Releases.
