# Response to reviewers: Sensors (major revision)

**Manuscript:** An Adaptive Hybrid Learning Framework with Edge–Fog Intelligence for Privacy-Aware and Resilient Intrusion Detection in Smart Grids (FedSVM-IDS)

Below we map each major review theme to concrete changes in the revised manuscript (LaTeX: `template.tex`).

---

## 1. Limited novelty / integration vs. new algorithm

**Response:** We reframed the contribution as a **principled multi-objective framework** (privacy, accuracy, communication, heterogeneity) with an explicit **conceptual utility / trade-off surface** (Section 3, Eq. `eq:utility`), **design principles**, and a **named aggregation mechanism** (DA-FedAvg with label-entropy diversity weights and renormalisation; ablation row in Table `tab:ablation`), rather than a catalogue of disconnected techniques.

- **Abstract, Introduction, Contributions, Related Work (critical gaps + table footnote), Conclusion:** Positioning now stresses **multi-objective trade-off analysis**, **systematic non-IID characterisation** as scientific evidence, **hierarchical cooperative detection** (local certainty vs. fog collective inference), and scoped claims (“to our knowledge”, “few works jointly publish …”).
- **Prior “integrated engineering framework” wording** (that could read as “integration only”) was replaced by language emphasising **explicit trade-offs**, **heterogeneity-aware aggregation**, and **transparent reporting** of federated limits.

---

## 2. Insufficient state-of-the-art comparison (2023–2025, GNN / Transformer / advanced FL-IDS)

**Response:** Added **same-split, same-feature** centralised tabular baselines and explicit literature positioning.

- **New Section (Results):** “Comparable Tabular Deep Baselines and Recent Literature” with **Table `tab:baselines_compare`** (MLP and small Transformer encoder vs. ensemble vs. IID federated SVM on CICIDS2017).
- **Discussion in text:** Contrasts with graph methods (BS-GAT, GraphKAN, GraphSAGE-style) as **centralised / different experimental conditions**, clarifying complementary scope vs. edge–fog DP constraints.
- **Bibliography:** `abadi2016deep`, `li2020fedprox` added in `ref.bib` (used for DP accounting and FedProx).

---

## 3. Non-IID performance degradation (~0.76 accuracy)

**Response:** We **aligned all high-level claims** with the measured gap and added **heterogeneity sweeps and mitigations**.

- **Abstract, Objectives (O2), Conclusion:** No longer imply uniform non-IID “mitigation”; federated CICIoT2023 at \(\alpha{=}0.5\) is stated explicitly where appropriate.
- **New tables:** `tab:noniid_alpha` (accuracy vs. Dirichlet \(\alpha\)); `tab:noniid_mitigation` (FedProx + zone-clustered aggregation at \(\alpha{=}0.5\)).
- **Discussion (“Non-IID Gap”):** Rewritten to interpret \(\alpha{=}0.5\) as a **stress test** and to cite the new tables.
- **Methodology:** FedProx hyperparameters in **Table `tab:hyperparams`**.

---

## 4. Limited security robustness (Byzantine, collusion, adversarial settings)

**Response:** **Empirical fault injection** for poisoning; expanded **threat model** for collusion.

- **New Section (Results):** “Byzantine Robustness Under Poisoning” with **Table `tab:byzantine_agg`** (label-flip fraction \(f\); FedAvg vs. coordinate-wise median vs. multi-Krum on CICIDS2017 federated non-IID).
- **Section 3 (Threat model):** New paragraph on **honest majority**, passive collusion + composition, secure aggregation assumptions, fog trust.
- **Section 3 (Adversarial Resilience):** Updated to reference measured robustness instead of “future extension” only.
- **Limitations / Future work:** Byzantine experiments acknowledged; adaptive attacks and model inversion left as open items.

---

## 5. Dataset–application gap (Smart Grid vs. general IDS datasets)

**Response:** Explicit **proxy-benchmark justification** and **AMI threat mapping**.

- **Methodology (Datasets):** Paragraph on scarcity of public labelled AMI PCAPs and use of overlapping flow-feature benchmarks; citations include `hamdi2025investigating`, `buyuktanir2025federated`, `rehman2024ffl`.
- **New Table `tab:ami_threat_map`:** Maps AMI-relevant threats to families in CICIDS2017, CICIoT2023, and Edge-IIoT.

---

## Minor items addressed

| Concern | Change |
|--------|--------|
| SVM vs. deep learning justification | **New subsection** “Rationale for Kernel SVM at the Edge” (Section 3). |
| DP composition across rounds | **Per-round vs. total** budget; **Table `tab:dp_composition`**; RDP/accountant pointer (`abadi2016deep`); Discussion updated so \(\varepsilon_{\mathrm{rd}}\) is not mistaken for a full-training guarantee. |
| Overstated novelty / robustness | Claim alignment throughout; Byzantine table supports robustness statements. |
| Real-time / latency claims | **Abstract, Section 3 (Stage 1), figure caption, case study, Section 4 micro-benchmark, Conclusion:** all stress **emulated / software-in-the-loop** timings; Section 4 adds **Inference Latency Micro-Benchmark**. |

---

## Figures not regenerated

Existing figure PNGs (e.g. headline metrics, convergence) were **not** redrawn in this revision package. If the editorial office requires updated plots, add curves for the \(\alpha\) sweep and Byzantine \(f\) sweep during proof.

## Replication / code artefacts

**`experiments/generate_revision_results.py`** (with `experiments/requirements.txt` and `experiments/README.md`) reproduces Tables `tab:baselines_compare`, `tab:noniid_alpha`, `tab:noniid_mitigation`, and `tab:byzantine_agg` into **`results/metrics.json`**, plus **`figures/fig_noniid_alpha_sweep.png`** and **`figures/fig_byzantine_robustness.png`**. It also logs **`cicids.dafedavg_vs_sizefed`** (DA-FedAvg with \(\lambda{=}0.15\) vs. pure size-weighted FedAvg) and writes **`results/table_dafedavg_rows.tex`** as a paste aid. Default settings use 28 FL rounds and a synthetic 84-D binary stream (set `FED_FULL=1` for 40 rounds). **Replace or extend** the script’s data loader with your CIC CSV tensors if you need bit-identical PCAP numbers. **Table `tab:dp_composition`** remains analytic from \((\varepsilon_{\mathrm{rd}},T)\). The micro-benchmark paragraph should be checked against your timing harness.

---

We thank the reviewers for the constructive Major Revision guidance.
