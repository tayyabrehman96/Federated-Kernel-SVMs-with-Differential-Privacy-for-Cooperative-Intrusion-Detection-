# Federated Kernel SVMs with Differential Privacy — cooperative IDS (MDPI *Sensors*)

[![Repository](https://img.shields.io/badge/code-GitHub-181717?logo=github)](https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-)

**Official code repository:** [github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-](https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-)

This workspace contains **LaTeX sources** (`template.tex`, `ref.bib`), **experiment scripts**, and **small result fragments** for the manuscript *Federated Kernel SVMs with Differential Privacy for Cooperative Intrusion Detection in Smart Meter Networks*. Documentation here is derived from those sources and the Python tree (full PDF review is not required to use the repo).

## Authors and citation

- **People and emails:** [AUTHORS.md](AUTHORS.md)
- **GitHub citation file:** [CITATION.cff](CITATION.cff) (update with the MDPI **DOI** after publication)

## What to push to Git (and what not to)

**Not everything on your laptop should be committed.** See **[REPOSITORY.md](REPOSITORY.md)** for a technical include/exclude list, training-mode summary, and first-push checklist.

## Datasets (three benchmarks in the paper)

| Paper benchmark | Official source (download) | Layout in this repo |
|-----------------|----------------------------|----------------------|
| **CICIDS2017** | [UNB / CIC — IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html) | `experiments/CICIDS2017/` |
| **CICIoT2023** | [UNB / CIC — IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | Default flow-CSV path: `experiments/CICIDS2023/` (override with `CIC_FLOW_BENCHMARK_DIR` for another extract or updated mirror) |
| **Edge-IIoT** (Edge-IIoTset) | Dataset landing: [IEEE DataPort — Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications); paper DOI: [10.1109/TII.2022.3155656](https://doi.org/10.1109/TII.2022.3155656) | `experiments/edge_iiot/data/` |

Details, licensing, and path overrides: **[DATASETS.md](DATASETS.md)**.

## Training / code entry points

| Goal | Command / files |
|------|-----------------|
| **Federated synthetic tables** (revision metrics, no raw CIC CSVs) | `cd experiments` → `pip install -r requirements.txt` → `python run_federated_revision_tables.py` |
| **XGBoost baseline on flow CSV folder** | `cd experiments` → install `xgboost` → `python run_cic_flow_xgboost_baseline.py` |
| **CICIDS2017** | `experiments/CICIDS2017_code/train_cicids2017.ipynb` |
| **More detail** | [experiments/README.md](experiments/README.md), [REPOSITORY.md](REPOSITORY.md) |

Outputs for the federated synthetic run: `results/metrics.json`, `results/table_*.tex`, figures under `figures/` (see `experiments/README.md`).

## LaTeX

Building the paper requires the **MDPI `Definitions/`** class bundle locally (often **not** redistributed on GitHub). Figures are expected under `figures/` per `template.tex`.

## Related files

- Reviewer response: `response_to_reviewers.md`
- Synthetic add-on experiments: `experiments/enhanced_synthetic_cyber_attack/`
