# Dataset layout (public corpora only)

Folders mirror what **training scripts expect** once you download official data. Naming follows the **MDPI Sensors** benchmarks: **CICIDS2017**, **CICIoT2023**, **Edge-IIoTset**.

## Official sources

| Benchmark | URL |
|-----------|-----|
| CICIDS2017 | [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CICIoT2023 | [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| Edge-IIoTset | [https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) · DOI [10.1109/TII.2022.3155656](https://doi.org/10.1109/TII.2022.3155656) |

## Expected local paths

1. **CICIDS2017** — `experiments/CICIDS2017/*.pcap_ISCX.csv` (day files). Notebook: `experiments/CICIDS2017_code/train_cicids2017.ipynb`.

2. **CICIoT2023 / flow-CSV baseline** — default CSV root `experiments/CICIDS2023/` or set **`CIC_FLOW_BENCHMARK_DIR`**. Trainer: `experiments/CICIDS2023_code/cicids2023_xgboost_trainer.py` (chunked read, dedup, numeric coercion — **no separate preprocessing bundle** is shipped).

3. **Edge-IIoT** — preprocessed flows under `experiments/edge_iiot/data/` per the **61-feature** protocol in the article. **Git** only keeps `data/.gitkeep`.

## Licensing

Host raw dumps on your machine or on **Zenodo**/Releases if the licence allows redistribution.
