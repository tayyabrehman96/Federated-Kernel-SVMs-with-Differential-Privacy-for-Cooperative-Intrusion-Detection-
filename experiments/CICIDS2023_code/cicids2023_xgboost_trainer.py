#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import gc
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import xgboost as xgb

# Repo-root convention: see experiments/dataset_paths.py and root README (dataset layout + CIC_FLOW_BENCHMARK_DIR).
try:
    from dataset_paths import CIC_FLOW_BENCHMARK_DIR
except ImportError:
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    CIC_FLOW_BENCHMARK_DIR = Path(os.environ.get(
        "CIC_FLOW_BENCHMARK_DIR",
        str(_here.parent / "CICIDS2023"),
    ))


def normalize_label_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    raw = name.replace(".pcap", "")
    if raw.startswith("BenignTraffic"):
        return "Benign"
    return "Attack"


def downcast_float32(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all").fillna(0)
    df = df.drop_duplicates()
    return df


def load_balanced_dataset(
    data_dir: str,
    cap_per_class: int = 200000,
    chunk_size: int = 200000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    buffers = {"Attack": [], "Benign": []}
    counts = {"Attack": 0, "Benign": 0}

    for idx, path in enumerate(files, start=1):
        label = normalize_label_from_filename(path)
        print(f"Loading {idx}/{len(files)}: {os.path.basename(path)} -> {label}")
        try:
            for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                for c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
                Xc = chunk.select_dtypes(include=[np.number])
                Xc = downcast_float32(clean_chunk(Xc))
                if Xc.empty:
                    continue
                m = len(Xc)
                indices = np.arange(m)
                rng.shuffle(indices)
                Xc = Xc.iloc[indices]
                need = cap_per_class - counts[label]
                if need <= 0:
                    continue
                take = min(need, m)
                part = Xc.iloc[:take].copy()
                buffers[label].append(part)
                counts[label] += take
                del chunk, Xc
                gc.collect()
        except Exception as e:
            print(f"  Skipped {os.path.basename(path)}: {e}")
            continue

    Xa = pd.concat(buffers["Attack"], ignore_index=True) if buffers["Attack"] else pd.DataFrame()
    Xb = pd.concat(buffers["Benign"], ignore_index=True) if buffers["Benign"] else pd.DataFrame()
    y = np.concatenate([np.ones(len(Xa), dtype=np.int32), np.zeros(len(Xb), dtype=np.int32)])
    X = pd.concat([Xa, Xb], ignore_index=True)
    print(f"Final dataset: X={X.shape}, y Attack={np.sum(y==1)}, Benign={np.sum(y==0)}")
    return X, pd.Series(y)


def main() -> None:
    data_dir = str(CIC_FLOW_BENCHMARK_DIR)
    X, y = load_balanced_dataset(data_dir, cap_per_class=200000, chunk_size=200000, random_state=42)
    feature_names = list(X.columns)
    # Optional standardization for tree is not necessary; but helps with some skewed features
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X.values)

    X_train, X_val, y_train, y_val = train_test_split(X, y.values, test_size=0.2, random_state=42, stratify=y.values)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.08,
        "min_child_weight": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "max_bin": 256,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(params, dtrain, num_boost_round=2000, evals=evals, early_stopping_rounds=100, verbose_eval=50)

    y_proba = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    thresholds = np.linspace(0.2, 0.8, 121)
    accs = [accuracy_score(y_val, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(accs))
    best_t = float(thresholds[best_idx])
    y_pred = (y_proba >= best_t).astype(int)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    print(f"\nBest threshold={best_t:.3f} | acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
    print("\n=== XGBoost validation report ===")
    print(classification_report(y_val, y_pred, target_names=["Benign", "Attack"], digits=4))

    # Save model and artifacts
    model_path = "xgb_cicids2023.json"
    booster.save_model(model_path)
    with open("xgb_best_threshold.txt", "w", encoding="utf-8") as f:
        f.write(f"{best_t}\n")
    with open("xgb_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"acc={acc:.6f}\nprec={prec:.6f}\nrec={rec:.6f}\nf1={f1:.6f}\n")

    # Feature importance (gain) plot
    gain_importance = booster.get_score(importance_type="gain")
    if not gain_importance:
        print("No feature importance available from model.")
        return
    # Ensure all features in list
    for name in feature_names:
        gain_importance.setdefault(name, 0.0)
    items = sorted(gain_importance.items(), key=lambda kv: kv[1], reverse=True)
    top_k = 25
    top_items = items[:top_k]
    names = [k for k, _ in top_items][::-1]  # reverse for horizontal plot
    vals = [v for _, v in top_items][::-1]

    plt.figure(figsize=(10, 8))
    bar_color = "#4C78A8"  # consistent professional blue
    plt.barh(names, vals, color=bar_color)
    plt.xlabel("Gain (importance)")
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    out_path = "xgb_feature_importance_gain.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved model to {model_path}, threshold/metrics to text files, and feature importance to {out_path}")


if __name__ == "__main__":
    main()


