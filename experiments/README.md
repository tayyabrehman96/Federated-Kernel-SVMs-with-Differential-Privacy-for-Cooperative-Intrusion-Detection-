# FedSVM-IDS revision: reproducible metrics

## Paper benchmarks (three datasets)

| Name in `template.tex` | Typical data folder | Notes |
|------------------------|---------------------|--------|
| CICIDS2017 | `CICIDS2017/` | Notebook: `CICIDS2017_code/train_cicids2017.ipynb` |
| CICIoT2023 | `CICIDS2023/` (default) | XGBoost: `CICIDS2023_code/cicids2023_xgboost_trainer.py`. Override path with `CIC_FLOW_BENCHMARK_DIR` for CIC-IoT-2024. |
| Edge-IIoT | `edge_iiot/data/` | See `edge_iiot/README.md` |

Full download and naming notes: **`../DATASETS.md`** (repository root).

## Quick start

```bash
cd experiments
pip install -r requirements.txt
# Optional: skip PyTorch (second baseline becomes alternate MLP)
set FED_NO_TORCH=1
python run_federated_revision_tables.py
```

Outputs:

- `../results/metrics.json`: all table statistics
- `../results/table_noniid_alpha_rows.tex`: LaTeX rows for the α sweep
- `../figures/fig_noniid_alpha_sweep.png`
- `../figures/fig_byzantine_robustness.png`

## PCAP-flow XGBoost (CICIoT2023 / CIC-IoT-2024 CSV tree)

From `experiments/` (install `xgboost` first):

```bash
set CIC_FLOW_BENCHMARK_DIR=optional\override\path
python run_cic_flow_xgboost_baseline.py
```

## Environment variables

| Variable | Effect |
|----------|--------|
| `FED_FULL=1` | 40 rounds, larger sample counts (closer to paper-scale FL) |
| `FED_NO_TORCH=1` | Skip PyTorch; tabular “Transformer” row uses a second sklearn MLP |
| `CIC_FLOW_BENCHMARK_DIR` | Directory of `*.csv` flows for `cicids2023_xgboost_trainer.py` (default: `experiments/CICIDS2023`) |

## Method note

The script implements **federated logistic regression** (84-D binary) as a transparent proxy for the edge classifier: same Dirichlet non-IID construction, FedAvg / FedProx / median / multi-Krum, label-flip attackers, and optional Gaussian noise on uploads. It does **not** ship CIC CSVs; point the data loader to local preprocessed tensors if you need exact PCAP replication.
