# Federated kernel SVMs with differential privacy for cooperative intrusion detection

**Repository:** [github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-](https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-)  
**Publication:** MDPI *Sensors* — *Federated Kernel SVMs with Differential Privacy for Cooperative Intrusion Detection in Smart Meter Networks* (add **DOI** when available).  
**People & citation file:** [AUTHORS.md](AUTHORS.md) · [CITATION.cff](CITATION.cff)

This repository is **code-first**: Python/notebooks, methodology figures, and `results/metrics.json`. It does **not** ship manuscript LaTeX/BibTeX, raw PCAP/CSV corpora, trained checkpoints, or preprocessing dumps. **Zenodo** (below) is the intended place for optional frozen bundles after you mint a DOI.

---

### 1. Problem

AMI-scale smart metering increases exposure to **DoS/DDoS**, **false data injection**, and **reconnaissance**, while operators need **privacy**, **low uplink**, and models that survive **non-IID** clients. Centralised IDS training on raw flows is often unacceptable; federated learning must be paired with **formal or quantified DP**, careful aggregation, and **frugal communication**.

---

### 2. Goals

| ID | Objective |
|----|-----------|
| **G1** | Collaborative training **without centralising raw flows**; Gaussian DP with **composed** budgets. |
| **G2** | Measure and mitigate **non-IID** effects (Dirichlet/partition stress); **DA-FedAvg**, **FedProx**, robust aggregators. |
| **G3** | **Stacked ensemble** over decision scores + **SHAP** for operator-facing explanations. |
| **G4** | **Reproducible scripts** + clear links to **public datasets** and optional **Zenodo/model** artefacts. |

---

### 3. Methodology (concise)

**Edge:** local **kernel SVM**; **DP-noisy** updates leave the meter. **Fog:** **DA-FedAvg** (size-weighted FedAvg when \(\lambda{=}0\)). **Cloud / ensemble tier:** heterogeneous learners fused on **scores**, not raw packets. Pipeline sketched below matches the MDPI narrative (full algorithmic detail in the paper).

![Three-tier stack: edge SVM + DP, fog DA-FedAvg, cloud ensemble](pm.png)

![Workflow: ingestion → preprocessing → partitioning → local training → DP aggregation → evaluation](Methodology_SM.jpg)

---

### 4. Main empirical results (paper)

**Table A — Headline detection (best reported configurations; IID / centralised where noted).**

| Benchmark | Setting | Accuracy | ROC-AUC | Notes |
|-----------|---------|----------|---------|--------|
| **CICIDS2017** | Stacked ensemble (proposed) | **0.9910** | **0.998** | vs. single learners ~0.9889–0.9903; vs. cited research baseline 0.9400 acc / 0.9500 F1 |
| **CICIoT2023** | Centralised XGBoost | **0.9807** | **0.995** | Strong central tabular baseline; **~22 pt** non-IID gap vs. federated prototypes at Dirichlet \(\alpha{=}0.5\) (see paper) |
| **Edge-IIoT** | Stacked ensemble (central) | **0.9634** | **0.9792** | Fed SVM IID **0.9410**; non-IID \(\alpha{=}0.5\) **0.8140** |

**Table B — CICIDS2017 centralised ensemble vs. strong trees (paper Table, abridged).**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| LightGBM | 0.9899 | 0.9897 | 0.9899 | 0.9898 |
| XGBoost | 0.9898 | 0.9895 | 0.9898 | 0.9896 |
| Random Forest | 0.9903 | 0.9899 | 0.9903 | 0.9899 |
| Extra Trees | 0.9889 | 0.9883 | 0.9889 | 0.9884 |
| **Ensemble (proposed)** | **0.9910** | **0.9904** | **0.9910** | **0.9905** |
| *Research baseline (cited)* | *0.9400* | *0.9440* | *0.9760* | *0.9500* |

**Table C — Edge-IIoT (paper Table, abridged).**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Central XGBoost | 0.9630 | 0.9701 | 0.9589 | 0.9645 | 0.9790 |
| Central LightGBM | 0.9612 | 0.9685 | 0.9568 | 0.9626 | 0.9771 |
| Central RF | 0.9598 | 0.9671 | 0.9541 | 0.9606 | 0.9758 |
| **Ensemble (proposed)** | **0.9634** | **0.9712** | **0.9598** | **0.9655** | **0.9792** |
| Fed SVM (IID) | 0.9410 | 0.9482 | 0.9351 | 0.9416 | 0.9541 |
| Fed SVM (non-IID) | 0.8140 | 0.8209 | 0.8074 | 0.8141 | 0.8832 |

**Synthetic replay in this repo:** `experiments/run_federated_revision_tables.py` updates **`results/metrics.json`** (non-IID \(\alpha\) sweep, Byzantine replay, etc.); numbers there are **simulation-scale** and complement—but do not replace—the full paper tables.

---

### 5. State of the art (positioning)

Recent **centralised** graph and deep IDS (e.g. **BS-GAT**, **GraphKAN**, large CNN/Transformer pipelines) often maximise offline accuracy when graphs/features are assembled in a data centre. **This work** targets a different operating point: **compact uplink** (84-D SVM-style messages vs. multi‑MB checkpoints), **formal DP composition** on uploads, **measured non-IID** stress, and a **fog/cloud ensemble over decision scores**. Direct numeric comparison across papers is often invalid (features, splits, binary vs. multi-class differ); treat cross-paper numbers as **directional**. The MDPI article’s related-work table maps prior SG/FL IDS along **DP/non-IID**, **fog ensemble**, **interpretability**, and **open gaps**.

---

### 6. Data, models, and Zenodo

| Artefact | Policy |
|----------|--------|
| **Raw benchmark CSV / PCAP-derived flows** | **Not in Git.** Download from official pages (below); keep on disk; set `CIC_FLOW_BENCHMARK_DIR` when needed. |
| **Preprocessing** (chunked load, dedup, scaling in trainers) | Implemented **inside** `cicids2023_xgboost_trainer.py` and notebooks — no separate “preprocessing dump” is uploaded. |
| **Trained boosters / checkpoints** (`.json`, large pickles) | **Not in Git.** Reproduce with `run_cic_flow_xgboost_baseline.py` / notebooks, or attach to a **release**. |
| **Zenodo (recommended)** | After acceptance, mint a DOI for optional:** frozen `metrics.json`**, **small config YAML**, **hashes of dataset splits**, or **permitted** model exports — e.g. `https://doi.org/10.5281/zenodo.XXXXXXX` *(replace with your upload)*. Link it here and in **About**. |
| **Hugging Face / other model hubs** | Optional mirror for edge distilled models; add the URL next to Zenodo when published. |

**Official dataset entry points**

| Corpus | URL |
|--------|-----|
| CICIDS2017 | [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CICIoT2023 | [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| Edge-IIoTset | [https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) |
| Edge-IIoT article | [https://doi.org/10.1109/TII.2022.3155656](https://doi.org/10.1109/TII.2022.3155656) |

More layout notes: [DATASETS.md](DATASETS.md). What belongs in Git: [REPOSITORY.md](REPOSITORY.md).

---

### 7. Quick start

```bash
git clone https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-.git
cd Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-/experiments
pip install -r requirements.txt
python run_federated_revision_tables.py   # refreshes results/metrics.json + local figures/ (not tracked)
```

XGBoost flow baseline: `python run_cic_flow_xgboost_baseline.py` (install `xgboost` first).

---

### 8. Discussion & future work

**Discussion:** Public IoT/IIoT corpora are **proxies** for proprietary AMI traffic; **DP** improves privacy but can **hurt calibration** for probability-based triage; **non-IID** remains the dominant federated bottleneck on CICIoT2023-style profiles.

**Future work:** utility **IEC/DNP3/Modbus WAN** captures; **async** FL; **personalised** aggregation; stronger **compression** for MCU-class meters; **Zenodo + HF** releases with versioned checkpoints when licenses allow.

---

### Suggested GitHub “About” text

`Federated kernel SVMs + DP + DA-FedAvg + ensemble for cooperative IDS (MDPI Sensors). Code + figures; datasets & models via linked corpora / Zenodo.`
