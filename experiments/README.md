# Experiments

Benchmark names follow the MDPI article: **CICIDS2017**, **CICIoT2023**, **Edge-IIoTset**. Official URLs and expected folder layout are in the **root [README.md](../README.md)** (§ Benchmarks).

| Benchmark | Data folder (default) | Entry point |
|-----------|----------------------|-------------|
| CICIDS2017 | `CICIDS2017/` | `CICIDS2017_code/train_cicids2017.ipynb` |
| CICIoT2023 | `CICIDS2023/` or `CIC_FLOW_BENCHMARK_DIR` | `run_cic_flow_xgboost_baseline.py` / `CICIDS2023_code/cicids2023_xgboost_trainer.py` |
| Edge-IIoT | `edge_iiot/data/` | (add trainer when available) |

## Federated synthetic driver

```bash
pip install -r requirements.txt
python run_federated_revision_tables.py
```

Writes **`../results/metrics.json`** and PNGs under **`../figures`**. Raster assets are not tracked on `main` except root `pm.png` and `Methodology_SM.jpg`.

## Environment variables

| Variable | Effect |
|----------|--------|
| `FED_FULL=1` | Longer / larger FL replay |
| `FED_NO_TORCH=1` | Skip PyTorch |
| `CIC_FLOW_BENCHMARK_DIR` | Absolute path to flow-CSV root for XGBoost baseline |
