#!/usr/bin/env python3
"""
Train the centralised XGBoost baseline on PCAP-flow CSVs (paper: CICIoT2023).

Runs inside `CICIDS2023_code/` so model/JSON/text artefacts land next to the
trainer, matching the existing layout. Override data with:

  set CIC_FLOW_BENCHMARK_DIR=path\to\cic-iot-flow-csv

From repo root: `python experiments/run_cic_flow_xgboost_baseline.py`
Or: `cd experiments && python run_cic_flow_xgboost_baseline.py`
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent / "CICIDS2023_code"


def main() -> None:
    os.chdir(_CODE)
    if str(_CODE) not in sys.path:
        sys.path.insert(0, str(_CODE))
    # Trainers import `dataset_paths` from parent `experiments/`
    exp_root = _CODE.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    import cicids2023_xgboost_trainer as trainer

    trainer.main()


if __name__ == "__main__":
    main()
