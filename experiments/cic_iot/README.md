# CIC-IoT (paper: CICIoT2023)

- **Official data:** [UNB — IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- This codebase names the default flow-CSV folder `CICIDS2023/` for the PCAP-flow XGBoost path; override with `CIC_FLOW_BENCHMARK_DIR` if your extract differs.

```bash
cd ../CICIDS2023_code
pip install xgboost pandas scikit-learn matplotlib numpy
python cicids2023_xgboost_trainer.py
```
