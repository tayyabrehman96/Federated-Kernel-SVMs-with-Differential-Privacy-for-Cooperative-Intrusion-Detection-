# CIC-IoT benchmark (manuscript: CICIoT2023)

- **Official data:** [UNB CIC — IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- **Paper naming:** `template.tex` uses *CICIoT2023*.
- **This clone:** PCAP-flow CSVs live in `experiments/CICIDS2023/`; XGBoost + reports in `experiments/CICIDS2023_code/`.
- **CIC-IoT-2024:** set environment variable `CIC_FLOW_BENCHMARK_DIR` to your CSV root (see root `DATASETS.md`).

Train the included baseline:

```bash
cd ../CICIDS2023_code
pip install xgboost pandas scikit-learn matplotlib numpy
python cicids2023_xgboost_trainer.py
```
