# Experiments

## Benchmarks (paper names)

| Benchmark | Typical data folder | Entry point |
|-----------|---------------------|-------------|
| CICIDS2017 | `CICIDS2017/` | `CICIDS2017_code/train_cicids2017.ipynb` |
| CICIoT2023 | `CICIDS2023/` or `CIC_FLOW_BENCHMARK_DIR` | `run_cic_flow_xgboost_baseline.py` / `CICIDS2023_code/cicids2023_xgboost_trainer.py` |
| Edge-IIoT | `edge_iiot/data/` | Add training script when wired to the paper pipeline |

Dataset URLs: **`../DATASETS.md`**.

## Federated synthetic replay

```bash
pip install -r requirements.txt
set FED_NO_TORCH=1   # optional
python run_federated_revision_tables.py
```

Writes **`../results/metrics.json`** and PNGs under **`../figures/`** (figures are git-ignored except root `pm.png` / `Methodology_SM.jpg`). **No `.tex` files** are produced.

## Environment variables

| Variable | Effect |
|----------|--------|
| `FED_FULL=1` | More rounds / samples |
| `FED_NO_TORCH=1` | Skip PyTorch |
| `CIC_FLOW_BENCHMARK_DIR` | Override flow-CSV root for XGBoost trainer |
