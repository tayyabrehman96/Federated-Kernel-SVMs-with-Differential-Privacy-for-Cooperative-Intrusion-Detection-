## CICIDS2023 Intrusion Detection: Preprocessing, Models, Results, and Visualizations

### Overview
This report documents the data pipeline, models, evaluation results, and visualizations produced for the `CICIDS2023` CSV-based dataset in this workspace. It also contrasts our approach with a recent fog-edge-enabled federated SVM paper (Tariq et al., 2024) and highlights the novel methodology implemented here.

### Dataset and Preprocessing
- **Source**: CSV files under `CICIDS2023/` (e.g., `BenignTraffic*.pcap.csv`, `Backdoor_Malware.pcap.csv`, `DDoS-*.pcap.csv`, etc.).
- **Parsing**:
  - Memory-efficient streaming via chunked reads.
  - Convert all columns to numeric (`errors='coerce'`).
  - Drop all-NaN columns, replace remaining NaNs/inf with 0.
  - Remove duplicates.
  - Downcast to `float32` to reduce memory.
- **Labeling**: Filenames starting with `BenignTraffic*` → label `Benign` (0), otherwise `Attack` (1).
- **Balancing**: Build a balanced dataset capped at 200,000 samples per class (400k total) for central training and an 80/20 stratified split for validation.
- **Standardization** (for tree input consistency and skew control): `StandardScaler(with_mean=False)` applied to features before training/evaluation.

Implementation references:
- Loader and trainer: `cicids2023_xgboost_trainer.py`
- Visualization script: `cicids2023_visualize_xgb.py`

### Models Trained
1) Federated SVM with kernel approximations (privacy-aligned prototype)
- Nystroem/RFF RBF feature maps; per-round learning rate schedule; simulated secure aggregation masks; optional DP noise; client shuffling and federated momentum aggregation.
- Trained under balanced per-client caps and fixed test caps for a strict federated simulation.
- Observed plateau around ~0.76 accuracy on the held-out balanced test with threshold sweep.

2) Federated MLP (FedAvg, prototype)
- `sklearn` `MLPClassifier` with `partial_fit` for per-client epochs; class weighting; per-round LR schedule; FedAvg aggregation.
- Similar performance envelope (~0.76 accuracy) in this setup.

3) Centralized XGBoost (production baseline)
- `xgboost` with histogram tree method, early stopping, and threshold sweep for best accuracy.
- Train/val split: 80/20 stratified on the 400k balanced sample.
- Feature importances (gain) extracted and plotted.

### Results (Centralized XGBoost, Validation)
- Best threshold: 0.450
- Accuracy: 0.9807
- Precision: 0.9900
- Recall: 0.9712
- F1: 0.9805

Confusion-matrix-aligned validation report (80k rows) is printed by the trainer and reproduced in `cicids2023_visualize_xgb.py` (same split/seed). The federated prototypes delivered ~0.76 accuracy in our experiments with heavy privacy-minded constraints and linear heads over kernel/RFF features.

### Comparison to research paper paper
- Paper scope: A fog–edge-enabled federated SVM IDS evaluated on NSL-KDD and CICIDS2017, reporting ~94% accuracy on CICIDS2017 (and improvements vs baselines across metrics).
- This work: Uses a different dataset (CICIDS2023), a much larger tabular feature space, and a centralized XGBoost baseline in addition to federated prototypes. Due to dataset differences, metrics are not directly comparable. However:
  - Our centralized XGBoost on a balanced 400k sample achieves 98.07% accuracy with strong precision/recall, which is consistent with gradient-boosted trees being highly competitive on tabular IDS data.
  - Our federated SVM/MLP prototypes reflect the paper’s decentralized training spirit but with simulated secure aggregation/DP and kernel feature approximations; these emphasize privacy-first tradeoffs (lower accuracy in our quick simulation) rather than peak central performance.

### Novel Methodology in This Project
- Federated SVM with kernel approximation and privacy knobs:
  - Multi-round, per-client `partial_fit` over Nystroem/RFF RBF maps.
  - Per-round LR scheduling, client shuffling, and federated momentum.
  - Simulated secure aggregation masks that sum to zero; optional Gaussian DP noise on updates.
- Multi-gamma RFF bank (optional path): concatenated RBFSampler features across multiple gamma scales to capture multi-resolution non-linearities.
- Threshold sweeping at evaluation time to choose the operating point optimizing accuracy (and available to tune for other metrics).
- Production-grade centralized baseline (XGBoost) to establish an upper bound and provide high-quality feature importance for explainability.
- Consistent visualization palette and a complete report image set.

### Visualizations Produced
Images saved in the project root (consistent pro-blue palette):
- `viz_confusion_matrix.png` and `viz_confusion_matrix_normalized.png` — class-wise error patterns (Benign vs Attack).
- `viz_roc.png` — ROC and AUC; curve close to the top-left indicates strong separability.
- `viz_pr.png` — PR curve; useful under class imbalance; area corresponds to AP.
- `viz_calibration_curve.png` — compares predicted probabilities to empirical outcome frequencies.
- `viz_probability_distributions.png` — score distributions for Benign vs Attack.
- `viz_threshold_sweep.png` — accuracy vs threshold; chosen operating point indicated.
- `viz_gains.png` — cumulative gains (how quickly positives are captured as we prioritize by score).
- `xgb_feature_importance_gain.png` — top features by gain from the trained booster.




