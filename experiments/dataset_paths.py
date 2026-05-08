"""
Canonical paths for the three MDPI paper benchmarks (see ../DATASETS.md).

Import this from experiment scripts so clones share one layout.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

# CICIDS2017 — day CSVs
CICIDS2017_DATA = EXPERIMENTS / "CICIDS2017"

# Flow-CSV tree used by XGBoost baseline; paper name CICIoT2023, folder historically CICIDS2023
_DEFAULT_FLOW_ROOT = EXPERIMENTS / "CICIDS2023"
CIC_FLOW_BENCHMARK_DIR = Path(
    os.environ.get("CIC_FLOW_BENCHMARK_DIR", str(_DEFAULT_FLOW_ROOT))
)

# Edge-IIoT — place preprocessed CSVs here
EDGE_IIOT_DATA = EXPERIMENTS / "edge_iiot" / "data"

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
