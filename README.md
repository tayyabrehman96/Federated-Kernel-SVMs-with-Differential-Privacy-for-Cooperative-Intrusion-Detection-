# Federated kernel SVMs with differential privacy for cooperative intrusion detection

**Repository:** [github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-](https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-)  
**Article (working title):** *Federated Kernel SVMs with Differential Privacy for Cooperative Intrusion Detection in Smart Meter Networks* — MDPI *Sensors* (add DOI here after acceptance).  
**Contacts & citation metadata:** [AUTHORS.md](AUTHORS.md) · [CITATION.cff](CITATION.cff)

This README summarises the **problem**, **objectives**, **methodology**, **empirical picture**, **discussion**, and **future work** at a level suitable for researchers and reviewers. The repository itself holds **source code**, **LaTeX/BibTeX**, two **reference figures** (`pm.png`, `Methodology_SM.jpg`), and **small numeric artefacts**; raw benchmark CSVs and the full figure set stay **out of version control** (download links below).

---

### 1. Problem

Advanced Metering Infrastructure (AMI) connects a very large number of endpoints to IP-based networks. That connectivity increases exposure to **denial-of-service** patterns, **false data injection (FDI)**, **reconnaissance**, and related threats, while operational constraints require **low uplink cost**, **formal or measurable privacy** for customer and grid-side traffic, and robustness when client data are **statistically heterogeneous (non-IID)**. Centralised training of intrusion detectors on all raw flows conflicts with privacy regulation and bandwidth limits; naive federated training, in turn, can diverge or leak information through gradients unless carefully designed.

---

### 2. Goals

- **G1 — Privacy-preserving collaboration:** Train cooperative detectors **without centralising raw flows**; quantify privacy via **Gaussian differential privacy** with **composed** budgets across rounds.  
- **G2 — Heterogeneity-aware federation:** Characterise **non-IID** clients (e.g. Dirichlet concentration sweeps), and employ **diversity-aware Federated Averaging (DA-FedAvg)**, **FedProx**, and robust aggregators where relevant.  
- **G3 — Accurate, interpretable decisions:** Combine **edge-side** margin scoring with **fog/cloud** fusion via a **stacked ensemble** and support operator triage with **SHAP**.  
- **G4 — Reproducibility:** Provide scripts that regenerate **table-ready metrics** and documented dataset **URLs** rather than opaque binary dumps.

---

### 3. Proposed solution (high level)

The manuscript develops an **edge–fog–cloud** cooperative IDS:

- **Edge:** Local **kernel SVM** (RBF) training on private metering/flow features; only **DP-noisy parameter updates** leave the device.  
- **Fog:** **DA-FedAvg** (reducing to size-weighted FedAvg when the diversity coefficient \(\lambda{=}0\)) aggregates edge updates into a **global** model.  
- **Cloud / fog stack:** A **stacked ensemble** consumes **decision scores** (not raw packets) for **multi-family** threat separation; **SHAP** explains contributing features.  

A two-stage **cooperative detection** view maps **volume-heavy** abuse to fast edge scoring and **correlation-sensitive** scenarios to fog-side reasoning over cross-client score structure, under emulated timing assumptions detailed in the paper.

**Architecture (stacked ensemble / three-tier view):**

![Three-tier cooperative IDS: edge SVM with DP updates, fog DA-FedAvg, cloud stacked ensemble](pm.png)

**End-to-end methodology pipeline (preprocessing → non-IID setup → federation → ensemble → evaluation):**

![Six-stage methodology workflow (ingestion, preprocessing, partitioning, local training, DP aggregation, ensemble & SHAP)](Methodology_SM.jpg)

---

### 4. Methodology (technical)

| Layer | Mechanism | Role |
|--------|-----------|------|
| Data | CICFlowMeter-style **tabular flows** (84-D on CICIDS2017; overlapping feature families on IoT/IIoT benchmarks) | Standardised comparison across **CICIDS2017**, **CICIoT2023**, **Edge-IIoTset** |
| Non-IID | **Dirichlet** client allocation; “zone-aware” stress regimes | Quantifies accuracy **gaps** between IID, mild, and strong skew |
| Local model | **Federated kernel SVM** proxy / **kernel SVM** centralised baselines (see paper for full algorithmic block) | Strong margins on medium-dimensional intrusion features |
| Privacy | **Gaussian mechanism** on uploaded statistics; **\((\varepsilon,\delta)\)** bookkeeping | Limits inference from repeated rounds |
| Aggregation | **FedAvg**, **DA-FedAvg**, **FedProx**, robust variants (e.g. coordinate-wise median, **multi-Krum**) where studied | Handles stragglers, attacks, and heterogeneity |
| Global decision | **Stacked ensemble** + calibration (e.g. **Platt** scaling under noise) | Improves F1/operating-range stability vs single learners |
| Explanation | **SHAP** on held-out samples | Triage and audit-facing attributions |

**Code in this repository (mapping):**

- **Federated simulation & LaTeX row generation** (synthetic IID/non-IID stress, DP, Byzantine replay): run `experiments/run_federated_revision_tables.py` after `pip install -r experiments/requirements.txt`.  
- **Flow-folder XGBoost baseline** (binary benign vs attack on PCAP-derived CSVs): `experiments/run_cic_flow_xgboost_baseline.py` with optional `CIC_FLOW_BENCHMARK_DIR`.  
- **CICIDS2017 notebook:** `experiments/CICIDS2017_code/train_cicids2017.ipynb`.  
- **Paper sources:** `template.tex`, `ref.bib`; small exports in `results/`.

Dataset **download URLs**, folder conventions, and license caution: **[DATASETS.md](DATASETS.md)**.  
Version-control policy (what is and is not tracked): **[REPOSITORY.md](REPOSITORY.md)**.

---

### 5. Results (summary)

Reported headline figures in the manuscript abstract (centralised or IID federated regimes, depending on the table) include **accuracies up to 99.10% (CICIDS2017), 98.07% (CICIoT2023), 96.34% (Edge-IIoT)** with **ROC-AUC up to 0.998 / 0.995 / 0.979**. Under **non-IID** replay on a CICIoT2023-style profile at Dirichlet concentration **0.5**, federated accuracy near **83%** vs a **87%** centralised ceiling illustrates the **residual generalisation gap** the paper analyses together with **FedProx**, clustered aggregation, and calibration. Exact tables, plots, and statistical **mean \(\pm\) std** over seeds appear in the PDF; this repo supplies the **scripts** and **JSON/TeX helpers** to regenerate the bundled numerical snippets.

---

### 6. Discussion

**Strengths:** The design **avoids raw-flow centralisation**, exposes **DP composition**, and reports **non-IID** stress tests on modern IoT/IIoT corpora rather than only IID splits. The **ensemble over score space** is communication-frugal relative to shipping full checkpoints each round. **SHAP** aligns with operational explainability requirements.

**Limitations (as framed in the paper):** Public PCAP corpora are **proxies** for proprietary AMI backhaul; **IEC 60870-5-104 / DNP3 / Modbus** WAN captures remain **future work**. **Federated kernel SVMs** trade off modelling flexibility vs communication; deep baselines are discussed as comparisons. **DP noise** can distort **probability calibration** even when thresholded accuracy is stable—an issue the manuscript highlights for alert triage.

---

### 7. Future work

- Protocol-specific evaluation on **utility SCADA/WAN** captures and AMI testbeds.  
- **Asynchronous** or **semi-asynchronous** federation under straggler-heavy AMI deployments.  
- **Personalised FL** and **clustered aggregation** with formal privacy–utility curves beyond the reported sweeps.  
- **Compression** of kernel expansions or distilled edge models for **MCU-class** meters.  
- Continued **release hygiene:** Zenodo DOI for frozen CSV subsets if license permits; CI checks on scripts.

---

### 8. Quick start (developers)

```bash
git clone https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-.git
cd Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-/experiments
pip install -r requirements.txt
python run_federated_revision_tables.py
```

Install XGBoost separately for the PCAP-flow baseline. Raw CSVs must be obtained from **[DATASETS.md](DATASETS.md)** and kept outside Git.

---

### 9. Key links (datasets & corpus pages)

| Resource | URL |
|----------|-----|
| CICIDS2017 | [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CICIoT2023 | [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| Edge-IIoTset (IEEE DataPort) | [https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) |
| Edge-IIoTset (article DOI) | [https://doi.org/10.1109/TII.2022.3155656](https://doi.org/10.1109/TII.2022.3155656) |

---

### Suggested GitHub “About” description

`Edge–fog–cloud IDS: federated kernel SVMs + DP + DA-FedAvg + ensemble; MDPI Sensors code & methodology figures — datasets via linked corpora.`
