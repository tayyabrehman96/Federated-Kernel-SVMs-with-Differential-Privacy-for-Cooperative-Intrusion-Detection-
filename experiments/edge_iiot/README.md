# Edge-IIoT (paper benchmark)

The MDPI paper reports results on **Edge-IIoT** alongside CICIDS2017 and CICIoT2023.

## Layout

Place preprocessed flow/feature CSV files (or the structure your training script expects) under:

`data/`

(`data/.gitkeep` keeps the folder in Git; large files stay local—see root `README.md` § Benchmarks.)

## Next step for full reproduction

Wire the same preprocessing and model stack as in the manuscript into a script here (e.g. `train_edge_iiot_baseline.py`) once the public Edge-IIoT extract is available on your machine. Until then, CICIDS2017 and the flow-CSV baseline cover most of the code paths in this repo.
