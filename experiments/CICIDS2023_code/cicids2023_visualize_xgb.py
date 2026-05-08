#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    accuracy_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb


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
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    rng = np.random.default_rng(random_state)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    buffers: Dict[str, List[pd.DataFrame]] = {"Attack": [], "Benign": []}
    counts: Dict[str, int] = {"Attack": 0, "Benign": 0}

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
    feature_names = list(X.columns)
    print(f"Final dataset: X={X.shape}, y Attack={np.sum(y==1)}, Benign={np.sum(y==0)}")
    return X, pd.Series(y), feature_names


# Palette: consistent professional blues
BLUE = "#4C78A8"
BLUE_LIGHT = "#7EAED1"
BLUE_DARK = "#2A5783"


def save_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, out_raw: str, out_norm: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    for mat, title, path, fmt in [
        (cm, "Confusion Matrix", out_raw, "d"),
        (cm_norm, "Confusion Matrix (Normalized)", out_norm, ".2f"),
    ]:
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Benign", "Attack"])
        plt.yticks(tick_marks, ["Benign", "Attack"])
        thresh = mat.max() / 2.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="white" if mat[i, j] > thresh else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()


def save_roc(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color=BLUE_LIGHT, lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_pr(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = np.trapz(precision[::-1], recall[::-1])
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color=BLUE, lw=2, label=f"AP ≈ {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_calibration(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=20, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color=BLUE_LIGHT)
    plt.plot(prob_pred, prob_true, marker="o", color=BLUE)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_probability_distributions(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(y_proba[y_true == 0], bins=50, alpha=0.6, color=BLUE_LIGHT, label="Benign")
    plt.hist(y_proba[y_true == 1], bins=50, alpha=0.6, color=BLUE, label="Attack")
    plt.xlabel("Predicted probability (Attack)")
    plt.ylabel("Count")
    plt.title("Probability Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    thresholds = np.linspace(0.0, 1.0, 201)
    accs = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        accs.append(accuracy_score(y_true, preds))
    best_idx = int(np.argmax(accs))
    best_t = float(thresholds[best_idx])
    best_acc = accs[best_idx]
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, accs, color=BLUE, lw=2)
    plt.axvline(best_t, color=BLUE_LIGHT, linestyle="--", label=f"Best t={best_t:.3f}, acc={best_acc:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Threshold Sweep (Accuracy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_gains_lift(y_true: np.ndarray, y_proba: np.ndarray, gains_path: str) -> None:
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    total_positives = np.sum(y_true)
    cum_positives = np.cumsum(y_sorted)
    population = np.arange(1, len(y_sorted) + 1)
    gains = cum_positives / total_positives
    pct_population = population / len(y_sorted)
    plt.figure(figsize=(7, 5))
    plt.plot(pct_population, gains, color=BLUE, lw=2, label="Cumulative Gains")
    plt.plot([0, 1], [0, 1], linestyle="--", color=BLUE_LIGHT, label="Baseline")
    plt.xlabel("Proportion of Samples (sorted by score)")
    plt.ylabel("Proportion of Positives Captured")
    plt.title("Gains Chart")
    plt.legend()
    plt.tight_layout()
    plt.savefig(gains_path, dpi=160)
    plt.close()


def save_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str = "Metrics") -> None:
    rep = classification_report(y_true, y_pred, target_names=["Benign", "Attack"], output_dict=True, zero_division=0)
    acc = rep["accuracy"] if "accuracy" in rep else accuracy_score(y_true, y_pred)
    rows = [
        ["Class", "Precision", "Recall", "F1", "Support"],
        ["Benign",
         f"{rep['Benign']['precision']:.4f}", f"{rep['Benign']['recall']:.4f}", f"{rep['Benign']['f1-score']:.4f}", f"{int(rep['Benign']['support'])}"],
        ["Attack",
         f"{rep['Attack']['precision']:.4f}", f"{rep['Attack']['recall']:.4f}", f"{rep['Attack']['f1-score']:.4f}", f"{int(rep['Attack']['support'])}"],
        ["Accuracy", "", "", f"{acc:.4f}", f"{int(rep['Benign']['support'] + rep['Attack']['support'])}"]
    ]
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.axis('off')
    tbl = ax.table(cellText=rows, loc='center', cellLoc='center', colLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    # Load model and best threshold
    model_path = "xgb_cicids2023.json"
    thr_path = "xgb_best_threshold.txt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if os.path.exists(thr_path):
        with open(thr_path, "r", encoding="utf-8") as f:
            try:
                best_t_saved = float(f.read().strip())
            except Exception:
                best_t_saved = 0.5
    else:
        best_t_saved = 0.5

    # Rebuild validation split to evaluate and plot consistently
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CICIDS2023"))
    X, y, feature_names = load_balanced_dataset(data_dir, cap_per_class=200000, chunk_size=200000, random_state=42)
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X.values)
    X_train, X_val, y_train, y_val = train_test_split(X, y.values, test_size=0.2, random_state=42, stratify=y.values)

    booster = xgb.Booster()
    booster.load_model(model_path)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    y_proba = booster.predict(dval)
    y_pred = (y_proba >= best_t_saved).astype(int)
    # also compute accuracy-optimal threshold for a second table
    thresholds = np.linspace(0.0, 1.0, 201)
    accs = [accuracy_score(y_val, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(accs))
    best_t_acc = float(thresholds[best_idx])
    y_pred_acc = (y_proba >= best_t_acc).astype(int)

    print("Validation report with saved threshold:")
    print(classification_report(y_val, y_pred, target_names=["Benign", "Attack"], digits=4))

    # Plots
    save_confusion_matrices(y_val, y_pred, "viz_confusion_matrix.png", "viz_confusion_matrix_normalized.png")
    save_roc(y_val, y_proba, "viz_roc.png")
    save_pr(y_val, y_proba, "viz_pr.png")
    save_calibration(y_val, y_proba, "viz_calibration_curve.png")
    save_probability_distributions(y_val, y_proba, "viz_probability_distributions.png")
    save_threshold_sweep(y_val, y_proba, "viz_threshold_sweep.png")
    save_gains_lift(y_val, y_proba, "viz_gains.png")
    # result tables as PNG
    save_metrics_table(y_val, y_pred, "viz_results_table_saved.png", title=f"Saved-threshold Results (t={best_t_saved:.3f})")
    save_metrics_table(y_val, y_pred_acc, "viz_results_table_accopt.png", title=f"Accuracy-optimal Results (t={best_t_acc:.3f})")

    print("Saved plots and results tables: confusion matrices, ROC, PR, calibration, probability distributions, threshold sweep, gains, and PNG tables.")


if __name__ == "__main__":
    main()


