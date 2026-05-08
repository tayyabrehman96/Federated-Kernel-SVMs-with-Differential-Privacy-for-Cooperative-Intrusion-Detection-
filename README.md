# Federated kernel SVMs with differential privacy for cooperative intrusion detection

**Code repository:** [github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-](https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-)  

**Publication:** MDPI *Sensors* — *Federated Kernel SVMs with Differential Privacy for Cooperative Intrusion Detection in Smart Meter Networks* (insert **DOI** when assigned).  

**Cite this repository:** [CITATION.cff](CITATION.cff) (GitHub “Cite this repository”).

---

## Authors and affiliations

| Author | Institution | Email |
|--------|-------------|--------|
| **Farrukh Aslam Khan** *(corresponding author)* | Center of Excellence in Information Assurance, Deanship of Scientific Research, King Saud University, Riyadh 11653, Saudi Arabia | [fakhan@ksu.edu.sa](mailto:fakhan@ksu.edu.sa) |
| **Tayyab Rehman** | Department of Information Engineering, Computer Science, and Mathematics, University of L’Aquila, 67100 L’Aquila, Italy | [tayyab.rehman@graduate.univaq.it](mailto:tayyab.rehman@graduate.univaq.it) |
| **Noshina Tariq** | Department of Artificial Intelligence and Data Science, National University of Computer and Emerging Sciences, Islamabad 44000, Pakistan | [noshina.tariq@isb.nu.edu.pk](mailto:noshina.tariq@isb.nu.edu.pk) |
| **Jalal Almuhtadi** | Department of Computer Science, College of Computer and Information Sciences (CCIS), King Saud University, Riyadh, Saudi Arabia | [jalal@ksu.edu.sa](mailto:jalal@ksu.edu.sa) |

**Software maintenance:** [Tayyab Rehman (@tayyabrehman96)](https://github.com/tayyabrehman96). For reproducibility or implementation questions, open a **GitHub Issue** on this repository or email the corresponding author as appropriate.

---

## Scope

This project distributes **research software**: Python modules, a Jupyter notebook, two methodology figures, and a compact **`results/metrics.json`** from the federated simulation driver. It does **not** version manuscript LaTeX, bibliography files, raw benchmark corpora, intermediate preprocessing exports, or large trained weights—those remain **local** or, if you choose to share them under licence, as **attachments on a GitHub Release** for this repository (see below).

---

## 1. Problem

Large-scale **Advanced Metering Infrastructure (AMI)** expands the attack surface (e.g. DoS/DDoS, false data injection, reconnaissance) while imposing **privacy**, **bandwidth**, and **heterogeneous data** constraints. Cooperative intrusion detection should avoid centralising raw flows, yet must remain accurate under **non-IID** clients and auditable privacy mechanisms.

---

## 2. Objectives

| ID | Objective |
|----|-----------|
| **O1** | **Privacy-aware federation:** Gaussian **differential privacy** on shared statistics with **composed** budgets across rounds. |
| **O2** | **Heterogeneity:** Quantify and mitigate **non-IID** partitions (Dirichlet controls); **DA-FedAvg**, **FedProx**, and robust aggregation where evaluated. |
| **O3** | **Decision quality & interpretability:** **Stacked ensemble** on decision scores; **SHAP** for attribution. |
| **O4** | **Open implementation:** Executable pipeline, documented **dataset URLs**, and a clear policy for **weights and binaries** on **GitHub only**.

---

## 3. Methodology

The following summarises the **edge–fog–cloud** pipeline described in the MDPI *Sensors* manuscript: how data are represented, how clients learn and share information under **differential privacy**, how the **fog** aggregates under **non-IID** data, and how a **global ensemble** and **SHAP** complete the decision stack. Symbols and hyper-parameters are defined fully in the paper; this section orients implementers and readers of the code.

### 3.1 System architecture

The framework follows a **three-tier** deployment aligned with smart-meter networks:

1. **Edge (smart meters / field gateways):** Each client holds a **private** tabular flow dataset (e.g. CICFlowMeter-style **numerical features**; **84** dimensions on CICIDS2017, with compatible feature pipelines on IoT and IIoT benchmarks). **Raw flow records do not leave the edge.** Only compact model updates (or scores derived from local models) are eligible for transmission upward.
2. **Fog (regional aggregator):** Collects **privacy-preserving** updates from many edge clients, runs **federated aggregation**, and redistributes a **global** model (or sufficient statistics). This tier is responsible for **heterogeneity-aware** weighting (DA-FedAvg) and can act as an intermediary for **score vectors** used at the ensemble tier.
3. **Cloud / ensemble tier:** Hosts a **stacked ensemble** of diverse classifiers that consume **decision scores** and meta-features—not raw packets—so uplink remains orders of magnitude smaller than centralising full feature matrices or PCAPs.

The schematic **Figure A** (`pm.png`) illustrates DP updates from meters to a regional aggregator using **DA-FedAvg**, then **aggregated edge scores** feeding a **stacked ensemble** for multi-family threat discrimination.

### 3.2 Data representation and preprocessing

Public benchmarks are used as **standardised tabular IDS corpora**: each record is a **labelled flow** with statistical and protocol-derived attributes. **Preprocessing** (ingestion, duplicate removal, stratified splits, scaling where applicable) is performed **inside** the training scripts or notebooks, consistent with the experimental protocol in the article. The goal is to align **train/validation/test** partitions and **feature dimensionality** across centralised, federated, and ablation runs so that reported gaps (e.g. non-IID) reflect the **federation protocol**, not ad-hoc data leakage.

### 3.3 Local learning at the edge

At each edge client, intrusion detection is implemented with **margin-based** models dominated by **kernel SVM** (RBF) principles in the paper: they offer strong performance on **medium-dimensional** numeric flows, **convex** local optimisation (more stable than deep models under class skew), and **interpretable** real-valued **scores** \(s(\mathbf{x})\) that can be composed downstream.

**Differential privacy (DP):** Before an update is transmitted, **Gaussian noise** is applied according to an \((\varepsilon, \delta)\) **Gaussian mechanism** with sensitivity derived from the **\(\ell_2\)** norm of the releasable vector (e.g. SVM weights), as detailed in the manuscript. **Multiple rounds** compose privacy budgets; the article reports **round-level** and **total** budget accounting. DP protects against inference from **observing uploads**, not against a compromised edge device.

### 3.4 Federation, DA-FedAvg, and non-IID data

**Standard FedAvg** aggregates client models with weights proportional to **local sample counts**. **Diversity-aware Federated Averaging (DA-FedAvg)** reweights clients using a **diversity coefficient** \(\lambda\) based on **local label entropy** (clients with more **mixed** class mass receive higher weight after renormalisation). Setting \(\lambda{=}0\) recovers **size-weighted FedAvg**. This mitigates **aggregation bias** when Dirichlet concentration is low and some clients see nearly single-class shards.

**Non-IID simulation:** Client partitions are drawn with a **Dirichlet** allocation over class indices (concentration \(\alpha\): smaller \(\alpha\) implies **stronger** skew). The repository’s driver `generate_revision_results.py` uses the same **per-class Dirichlet** construction with **federated logistic regression** (84-D binary) as a **transparent, reproducible proxy** for the edge learner when generating `metrics.json`, **FedAvg / FedProx / median / multi-Krum** options, and label-flip **Byzantine** fractions—mirroring the **non-IID and robustness** narrative of the paper without shipping raw CIC CSVs.

**FedProx:** A proximal term penalises deviation from the **global** iterate on each client, improving stability when local objectives are badly conditioned under skew.

### 3.5 Cooperative detection and the stacked ensemble

The **cooperative detection** design separates:

- **Edge-side “local certainty”:** fast **margin/score** from the federated kernel SVM for **volume-oriented** or locally separable abuse; and  
- **Fog-side “global correlation”:** inference over **cross-client score patterns** (e.g. correlated anomalies across meters) without recentralising raw flows.

At the **ensemble tier**, several **heterogeneous** learners (e.g. boosting, random forests, extra trees) are **stacked** on **meta-inputs** derived from base-model **scores**, improving **multi-family** discrimination and calibration compared to any single base learner. **SHAP (SHapley Additive exPlanations)** is applied to explain **which flow features** drive alerts, supporting operator triage and audit.

### 3.6 Communication footprint (conceptual)

Compared with uploading **full feature tensors** or **raw PCAP** to a central IDS, transmitting **low-dimensional** model parameters (e.g. on the order of **tens to hundreds of floats per client per round** for linearised or kernelised surrogates in the paper’s configuration) yields a **large reduction in uplink**. Exact byte counts and round budgets appear in the article; the important design constraint is **no raw-flow centralisation** while keeping aggregation **practical** on AMI-class backhaul.

### 3.7 Figures

![Three-tier architecture: edge SVM with DP, regional DA-FedAvg, cloud stacked ensemble](pm.png)

![End-to-end workflow: ingestion, preprocessing, partitioning, local training, DP aggregation, evaluation](Methodology_SM.jpg)

**Figure B** (`Methodology_SM.jpg`) aligns with the **workflow** figure in the paper: ingestion → preprocessing → **Dirichlet** partitioning → **local training** → **DP aggregation (DA-FedAvg)** → **global ensemble & SHAP evaluation**.

---

## 4. Main reported results (from the paper)

**Table 1 — Headline performance (centralised or IID federated where noted).**

| Benchmark | Configuration | Accuracy | ROC-AUC | Comment |
|-----------|----------------|----------|---------|---------|
| CICIDS2017 | Stacked ensemble (proposed) | **0.9910** | **0.998** | Outperforms single learners; large margin vs. cited baseline (0.94 / F1 0.95) |
| CICIoT2023 | Centralised XGBoost | **0.9807** | **0.995** | ~**22** percentage-point non-IID gap under Dirichlet \(\alpha{=}0.5\) vs. federated prototypes (paper) |
| Edge-IIoT | Stacked ensemble (central) | **0.9634** | **0.9792** | Fed SVM IID **0.9410**; non-IID \(\alpha{=}0.5\) **0.8140** |

**Table 2 — CICIDS2017: ensemble vs. gradient-boosting / forests (abridged).**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| LightGBM | 0.9899 | 0.9897 | 0.9899 | 0.9898 |
| XGBoost | 0.9898 | 0.9895 | 0.9898 | 0.9896 |
| Random Forest | 0.9903 | 0.9899 | 0.9903 | 0.9899 |
| Extra Trees | 0.9889 | 0.9883 | 0.9889 | 0.9884 |
| **Ensemble (proposed)** | **0.9910** | **0.9904** | **0.9910** | **0.9905** |
| *External baseline (cited)* | *0.9400* | *0.9440* | *0.9760* | *0.9500* |

**Table 3 — Edge-IIoT (abridged).**

| Model | Acc. | Prec. | Rec. | F1 | AUC |
|-------|------|-------|------|-----|-----|
| Central XGBoost | 0.9630 | 0.9701 | 0.9589 | 0.9645 | 0.9790 |
| Central LightGBM | 0.9612 | 0.9685 | 0.9568 | 0.9626 | 0.9771 |
| Central RF | 0.9598 | 0.9671 | 0.9541 | 0.9606 | 0.9758 |
| **Ensemble (proposed)** | **0.9634** | **0.9712** | **0.9598** | **0.9655** | **0.9792** |
| Fed SVM (IID) | 0.9410 | 0.9482 | 0.9351 | 0.9416 | 0.9541 |
| Fed SVM (non-IID) | 0.8140 | 0.8209 | 0.8074 | 0.8141 | 0.8832 |

The driver **`experiments/run_federated_revision_tables.py`** refreshes **`results/metrics.json`** for the **synthetic** federated replay (non-IID sweep, Byzantine settings, etc.); interpret those entries as complementary to the **full experimental section** in the article.

---

## 5. Positioning relative to recent literature

State-of-the-art **centralised** graph and deep networks (graph attention, hybrid KAN variants, large convolutional or Transformer IDS) often prioritise **offline accuracy** when full graphs or massive batches are available in a data centre. **This work** targets **constrained uplink** (compact SVM-style updates versus multi-megabyte checkpoints), **explicit DP accounting**, **documented non-IID stress**, and a **multi-learner ensemble in score space**. Cross-paper metrics are **not** directly comparable without aligning features, splits, and task definitions; use the article’s related-work synthesis for structured comparison.

---

## 6. Benchmarks: official sources and local layout

Download each corpus **only from its official provider** and comply with its **licence**. This repository lists **URLs** and the **directory names** expected by the training scripts after you unzip or convert files locally.

| Benchmark | Provider & download | Typical local path (after download) | Entry point in this repo |
|-----------|------------------------|--------------------------------------|---------------------------|
| **CICIDS2017** | [UNB CIC — IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) | `experiments/CICIDS2017/*.pcap_ISCX.csv` | `experiments/CICIDS2017_code/train_cicids2017.ipynb` |
| **CICIoT2023** | [UNB CIC — IoT 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | Default `experiments/CICIDS2023/` or any path via **`CIC_FLOW_BENCHMARK_DIR`** | `experiments/run_cic_flow_xgboost_baseline.py` / `CICIDS2023_code/cicids2023_xgboost_trainer.py` |
| **Edge-IIoTset** | [IEEE DataPort — Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) · companion article [DOI 10.1109/TII.2022.3155656](https://doi.org/10.1109/TII.2022.3155656) | `experiments/edge_iiot/data/` (preprocessed **61-feature** flows per article) | Add your training script when wired to this dataset |

**Ingestion note:** For the PCAP-flow XGBoost path, **preprocessing** (chunked CSV read, deduplication, numeric coercion) is implemented **inside** `cicids2023_xgboost_trainer.py`; no separate preprocessing archive is required beyond the official dataset files.

---

## 7. Weights, binaries, and GitHub distribution policy

| Item | Policy |
|------|--------|
| **Default branch** | Source code, notebooks, `pm.png`, `Methodology_SM.jpg`, `results/metrics.json`, documentation within the tree. |
| **Raw CSV / PCAP-derived corpora** | **Excluded** from Git (size + licence). Obtain from the **official URLs** in §6. |
| **Trained checkpoints** (e.g. XGBoost `.json`, large pickles) | **Excluded** from ordinary commits (see `.gitignore`). Regenerate with `run_cic_flow_xgboost_baseline.py` and the notebook pipeline. |
| **Optional sharing of weights or summarized artefacts** | Use **[GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)** on **this** repository (upload zip under a version tag, describe licence and file checksums in the release notes). Everything stays under **GitHub**; no external archive is required by this policy. |

---

## 8. Quick start

```bash
git clone https://github.com/tayyabrehman96/Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-.git
cd Federated-Kernel-SVMs-with-Differential-Privacy-for-Cooperative-Intrusion-Detection-/experiments
pip install -r requirements.txt
python run_federated_revision_tables.py
```

Optional **`FED_NO_TORCH=1`** if PyTorch is not installed. For the flow-folder baseline, `pip install xgboost pandas` and run **`python run_cic_flow_xgboost_baseline.py`**. Further script-level options: **`experiments/README.md`**.

---

## 9. Discussion and future work

**Discussion:** Public IoT/IIoT datasets approximate—not replace—operator AMI traffic. **DP** protects uploads but can **flatten** probability calibration for triage. **Non-IID** skew remains the hardest federated failure mode on IoT-scale corpora.

**Future work:** Protocol-specific **IEC 60870 / DNP3 / Modbus** captures; **asynchronous** federation; **personalised** or **cluster-specialised** aggregation; stronger **model compression** for resource-constrained meters; optional **GitHub Release** bundles with frozen weights when licensing permits.

---

## Suggested GitHub “About” description

`Research code: federated kernel SVMs, differential privacy, DA-FedAvg, and ensemble IDS for smart meter networks (MDPI Sensors). Datasets via UNB CIC & IEEE DataPort; weights optional via Releases.`
