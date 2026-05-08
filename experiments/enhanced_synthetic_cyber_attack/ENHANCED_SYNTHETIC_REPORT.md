## Enhanced Synthetic Cyber Attack IDS: Approach, Preprocessing, Models, Results, and Visualizations

### Scope
This document summarizes the end‑to‑end IDS pipeline we built for the enhanced synthetic cyber attack dataset:
preprocessing, feature engineering, model training/evaluation, thresholding strategy, and research‑grade visualizations. It also contrasts our approach with federated SVM work (e.g., Tariq et al., 2024) and explains what each chart conveys.

### Data and Preprocessing
- Dataset: `enhanced_synthetic_cyber_attack/Enhanced_Synthetic_Cyber_Attack_Dataset.csv`
- Cleaning:
  - Convert non‑numeric to numeric where applicable and keep numeric columns
  - Replace inf/NaN → 0; drop all‑NaN columns; drop duplicates
  - Encode `Protocol` to `Protocol_code` via factorization; drop string identifiers likely to leak (e.g., `Timestamp`, `Source_IP`, `Destination_IP`, `Protocol`, `Attack_Type`)
- Target: `Is_Attack` (1 = attack, 0 = benign)
- Feature engineering (adds discriminative signal):
  - Ratios/interactions: `loss_per_size`, `loss_per_throughput`, `latency_x_jitter`, `duration_x_throughput`, `size_per_duration`
  - Log transforms for skewed features: `log1p_{Packet_Size, Duration, Packet_Loss, Latency, Throughput, Jitter}`
  - High‑value flags (75th percentile): `high_Latency`, `high_Jitter`

### Modeling Approach
- Primary split model: XGBoost (histogram tree method) with early stopping and a small hyperparameter sweep; train/val stratified split (80/20). On training folds we upsample minority (Attack) to 1:1.
- Cross‑validation model: LightGBM 5‑fold stratified CV (with Attack upsampling on each training fold) to provide a robust paper‑style summary independent of a single split.
- Threshold selection and operating points:
  - Accuracy‑optimal: maximizes overall accuracy
  - F1‑optimal: balances precision/recall for a single “balanced” operating point
  - Recall‑targeted: picks the smallest threshold meeting a target recall (we used 0.85) to mimic IDS detection priority (catch attacks)

### Novel Methodology in This Work
- Multi‑objective evaluation via distinct operating points (accuracy‑optimal, F1‑optimal, and recall‑targeted) to transparently show security trade‑offs.
- Attack‑aware training using simple upsampling on training folds to mitigate class imbalance, plus engineered features tailored to network metrics (ratios, interactions, logs, high‑value flags).
- Full visualization suite with a consistent professional blue palette: ROC, PR, calibration, probability distributions, threshold sweep, DET, KS, and per‑operating‑point confusion matrices.
- CV summary (LightGBM) reported to avoid single‑split bias and to better reflect expected variance on small/imbalanced data.

### Results Summary
Note: The dataset is small (~1k rows after cleaning) and imbalanced (~20% Attack), which limits separability. We therefore present multiple operating points.

Split‑based XGBoost (80/20 stratified):
- Accuracy‑optimal operating point (highest accuracy):
  - Accuracy ≈ 0.805
  - Attack: Precision ≈ 0.667, Recall ≈ 0.050, F1 ≈ 0.093
  - Interpretation: strong overall accuracy but misses most attacks (very conservative threshold)

- F1‑optimal operating point (balanced):
  - Accuracy ≈ 0.200 (threshold shifts very low on this small validation set)
  - Attack: Recall ≈ 1.0, but precision is low; benign class collapses (many false alarms)
  - Interpretation: catches attacks but not practical due to high false alarms; demonstrates the opposite extreme of the trade‑off

- Recall‑targeted operating point (target recall = 0.85):
  - Similar to F1‑optimal here due to flat curves in small data: high recall, low precision/accuracy
  - Interpretation: a “detection‑first” point; useful for reporting but requires post‑processing/triage in practice

5‑fold CV (LightGBM; upsampled Attack on training folds):
- ROC‑AUC ≈ 0.52, PR‑AUC ≈ 0.21 (modest separability consistent with small feature set)
- Accuracy‑optimal threshold: Accuracy ≈ 0.80 but Attack recall very low
- F1‑optimal and recall‑targeted thresholds: high recall but low precision/accuracy

Overall takeaway: On this small, engineered synthetic dataset, accuracy‑optimal looks best numerically but sacrifices Attack recall. Detection‑oriented thresholds recover recall at the expense of false alarms. Presenting both is best practice for IDS.

### Comparison with Federated SVM (Tariq et al., 2024)
- Paper (fog‑edge federated SVM): decentralized training, privacy preservation, and evaluation on NSL‑KDD and CICIDS2017; reports high accuracy/recall using a federated approach with SVM and fog computing.
- Our work (this synthetic dataset): centralized gradient boosting (XGBoost/LightGBM), plus engineered features and class upsampling; and explicit thresholding for different IDS objectives.
- Alignment: We adopt the paper’s evaluation spirit—showing accuracy/precision/recall/F1 and emphasizing operating points. We also previously prototyped federated SVM/MLP with secure aggregation and DP for CICIDS data in this project; however, for this small synthetic dataset we prioritized robust classical ML baselines and transparent op‑point reporting.
- Key difference: centralized boosted‑tree baselines vs decentralized SVM; our results are on a different (synthetic) dataset with distinct class balance and feature space, so direct metric matching is not appropriate. Methodologically, we contribute a clear operating‑point narrative and comprehensive visualization suite to aid research reporting.

### Visualizations a
Saved under `enhanced_synthetic_cyber_attack/`:
- `enh_roc.png`: ROC curve and AUC; shows true‑positive vs false‑positive trade‑off across thresholds. AUC near 0.5 indicates weak separation.
- `enh_pr.png`: Precision‑Recall curve and area; more informative under imbalance. Low PR‑AUC reflects difficulty distinguishing attacks from benign with current features/size.
- `enh_calibration.png`: Reliability of predicted probabilities; deviations from the diagonal indicate under/over‑confidence.
- `enh_probability_distributions.png`: Score histograms for benign vs attack; overlap indicates difficulty in separation.
- `enh_threshold_sweep.png`: Accuracy and F1 vs threshold; used to select accuracy‑optimal and F1‑optimal operating points.
- `enh_det.png`: DET curve (FPR vs FNR); visualizes trade‑off at low error regions.
- `enh_ks.png`: KS curve; gap between positive/negative CDFs indicates separability.
- Confusion matrices per operating point:
  - `enh_confusion_matrix_accopt*.png`: accuracy‑optimal classification errors
  - `enh_confusion_matrix_f1*.png`: balanced (F1‑optimal) errors
  - `enh_confusion_matrix_recall*.png`: recall‑targeted errors
- `enh_feature_importance_gain.png`: Top features by gain (XGBoost); highlights which engineered/native features the model relies on.

### Reproducibility
- Script: `enhanced_synthetic_cyber_attack/train_and_visualize_enhanced_synthetic.py`
- It performs preprocessing + feature engineering, LightGBM 5‑fold CV summary, XGBoost single‑split training, and saves all plots/metrics.




