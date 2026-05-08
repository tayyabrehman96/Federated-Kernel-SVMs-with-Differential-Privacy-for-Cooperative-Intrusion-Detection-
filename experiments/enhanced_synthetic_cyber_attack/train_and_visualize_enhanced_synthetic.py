#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

DATA_PATH = os.path.join(os.path.dirname(__file__), "Enhanced_Synthetic_Cyber_Attack_Dataset.csv")

# Consistent professional blue palette
BLUE = "#4C78A8"
BLUE_LIGHT = "#7EAED1"
BLUE_DARK = "#2A5783"


def load_and_preprocess(path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(path)
    # Ensure expected columns exist
    assert "Is_Attack" in df.columns, "Target column 'Is_Attack' not found"

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all").fillna(0)
    df = df.drop_duplicates()

    # Encode protocol
    if "Protocol" in df.columns:
        df["Protocol_code"], _ = pd.factorize(df["Protocol"].astype(str))

    # Drop string-heavy identifiers to avoid overfitting/leakage
    drop_cols = [c for c in ["Timestamp", "Source_IP", "Destination_IP", "Protocol", "Attack_Type"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Target
    y = df["Is_Attack"].astype(int)
    X = df.drop(columns=["Is_Attack"], errors="ignore")

    # Feature engineering (ratios, interactions, logs)
    def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
        return (a.astype(float)) / (b.astype(float) + eps)

    if all(col in df.columns for col in ["Packet_Loss", "Packet_Size"]):
        X["loss_per_size"] = safe_div(df["Packet_Loss"], df["Packet_Size"])
    if all(col in df.columns for col in ["Packet_Loss", "Throughput"]):
        X["loss_per_throughput"] = safe_div(df["Packet_Loss"], df["Throughput"])
    if all(col in df.columns for col in ["Latency", "Jitter"]):
        X["latency_x_jitter"] = (df["Latency"].astype(float)) * (df["Jitter"].astype(float))
    if all(col in df.columns for col in ["Duration", "Throughput"]):
        X["duration_x_throughput"] = (df["Duration"].astype(float)) * (df["Throughput"].astype(float))
    if all(col in df.columns for col in ["Packet_Size", "Duration"]):
        X["size_per_duration"] = safe_div(df["Packet_Size"], df["Duration"])

    # log1p transforms for skewed metrics
    for col in ["Packet_Size", "Duration", "Packet_Loss", "Latency", "Throughput", "Jitter"]:
        if col in df.columns:
            X[f"log1p_{col}"] = np.log1p(df[col].astype(float).clip(lower=0))

    # high/low indicators (75th percentile)
    for col in ["Latency", "Jitter"]:
        if col in df.columns:
            thr = float(np.percentile(df[col].astype(float), 75))
            X[f"high_{col}"] = (df[col].astype(float) > thr).astype(int)

    feature_names = list(X.columns)
    return X, y, feature_names


def upsample_minority(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y
    if len(pos_idx) >= len(neg_idx):
        return X, y
    # Upsample positives to match negatives
    extra = rng.choice(pos_idx, size=(len(neg_idx) - len(pos_idx)), replace=True)
    X_up = np.vstack([X, X[extra]])
    y_up = np.concatenate([y, y[extra]])
    # shuffle
    order = rng.permutation(len(y_up))
    return X_up[order], y_up[order]


def train_xgb(X: pd.DataFrame, y: pd.Series, feature_names: List[str], target_recall: float | None = None):
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
    )

    # Upsample Attack class on training set to 1:1 ratio
    X_train, y_train = upsample_minority(X_train, y_train, random_state=42)

    # imbalance handling
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = 1.0  # after upsampling

    # small hyperparameter search on validation set
    grid = [
        {"max_depth": d, "eta": lr, "min_child_weight": mcw, "subsample": ss, "colsample_bytree": cs}
        for d in (4, 6, 8)
        for lr in (0.05, 0.08, 0.12)
        for mcw in (1, 2, 4)
        for ss in (0.8, 1.0)
        for cs in (0.8, 1.0)
    ]
    best_booster = None
    best_f1 = -1.0
    best_cfg = None

    for cfg in grid:
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "tree_method": "hist",
            "max_depth": cfg["max_depth"],
            "eta": cfg["eta"],
            "min_child_weight": cfg["min_child_weight"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "lambda": 1.0,
            "alpha": 0.0,
            "max_bin": 256,
            "scale_pos_weight": scale_pos_weight,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        evals = [(dtrain, "train"), (dval, "val")]
        booster = xgb.train(params, dtrain, num_boost_round=600, evals=evals, early_stopping_rounds=50, verbose_eval=False)
        y_proba_tmp = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        thresholds = np.linspace(0.05, 0.95, 181)
        f1s = [f1_score(y_val, (y_proba_tmp >= t).astype(int), zero_division=0) for t in thresholds]
        f1_best = float(np.max(f1s))
        if f1_best > best_f1:
            best_f1 = f1_best
            best_booster = booster
            best_cfg = params

    booster = best_booster
    y_proba = booster.predict(xgb.DMatrix(X_val, label=y_val, feature_names=feature_names), iteration_range=(0, booster.best_iteration + 1))

    # Threshold optimization for accuracy and F1
    thresholds = np.linspace(0.0, 1.0, 1001)
    preds = np.array([(y_proba >= t).astype(int) for t in thresholds])
    recalls = np.array([recall_score(y_val, p, zero_division=0) for p in preds])
    accs = np.array([accuracy_score(y_val, p) for p in preds])
    f1s = np.array([f1_score(y_val, p, zero_division=0) for p in preds])

    # Accuracy- and F1-optimal thresholds
    t_acc = float(thresholds[int(np.argmax(accs))])
    t_f1 = float(thresholds[int(np.argmax(f1s))])

    if target_recall is None:
        used_t = t_f1
    else:
        # Target-recall threshold (smallest t that meets recall target); fallback to max recall
        valid_idx = np.where(recalls >= target_recall)[0]
        if len(valid_idx) > 0:
            used_t = float(thresholds[int(valid_idx[0])])
        else:
            used_t = float(thresholds[int(np.argmax(recalls))])

    # Use selected threshold for final preds
    y_pred_f1 = (y_proba >= used_t).astype(int)
    y_pred_acc = (y_proba >= t_acc).astype(int)
    # Recall-targeted threshold
    t_rec = None
    if target_recall is not None:
        valid_idx = np.where(recalls >= target_recall)[0]
        t_rec = float(thresholds[int(valid_idx[0])]) if len(valid_idx) > 0 else float(thresholds[int(np.argmax(recalls))])
    y_pred_rec = (y_proba >= (t_rec if t_rec is not None else used_t)).astype(int)

    # Curves
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    pr_prec, pr_rec, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = np.trapz(pr_prec[::-1], pr_rec[::-1])

    metrics = {
        "best_threshold_accuracy": t_acc,
        "best_threshold_f1": t_f1,
        "used_threshold_f1": used_t,
        "acc_at_f1": accuracy_score(y_val, y_pred_f1),
        "prec_at_f1": precision_score(y_val, y_pred_f1, zero_division=0),
        "rec_at_f1": recall_score(y_val, y_pred_f1, zero_division=0),
        "f1_at_f1": f1_score(y_val, y_pred_f1, zero_division=0),
        "acc_at_accopt": accuracy_score(y_val, y_pred_acc),
        "prec_at_accopt": precision_score(y_val, y_pred_acc, zero_division=0),
        "rec_at_accopt": recall_score(y_val, y_pred_acc, zero_division=0),
        "f1_at_accopt": f1_score(y_val, y_pred_acc, zero_division=0),
        "recall_target": target_recall,
        "used_threshold_recall": t_rec if t_rec is not None else used_t,
        "acc_at_recall": accuracy_score(y_val, y_pred_rec),
        "prec_at_recall": precision_score(y_val, y_pred_rec, zero_division=0),
        "rec_at_recall": recall_score(y_val, y_pred_rec, zero_division=0),
        "f1_at_recall": f1_score(y_val, y_pred_rec, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "config": best_cfg,
    }

    # Save artifacts
    model_path = os.path.join(os.path.dirname(__file__), "enh_synth_xgb.json")
    booster.save_model(model_path)
    with open(os.path.join(os.path.dirname(__file__), "enh_synth_thresholds.txt"), "w", encoding="utf-8") as f:
        f.write(f"t_acc={t_acc}\n")
        f.write(f"t_f1={t_f1}\n")
        if t_rec is not None:
            f.write(f"t_recall_target={t_rec}\n")
    with open(os.path.join(os.path.dirname(__file__), "enh_synth_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}={v}\n")

    return booster, (X_val, y_val, y_proba, y_pred_f1, y_pred_acc, y_pred_rec), metrics


def plot_confusion_matrices(y_true, y_pred, out_raw, out_norm):
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


def plot_roc(y_true, y_proba, out_path):
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


def plot_pr(y_true, y_proba, out_path):
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


def plot_det(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fnr = 1 - tpr
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, fnr, color=BLUE, lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_ks(y_true, y_proba, out_path):
    # KS statistic between positive and negative score CDFs
    scores_pos = np.sort(y_proba[y_true == 1])
    scores_neg = np.sort(y_proba[y_true == 0])
    cdf_pos = np.arange(1, len(scores_pos) + 1) / len(scores_pos) if len(scores_pos) > 0 else np.array([0])
    cdf_neg = np.arange(1, len(scores_neg) + 1) / len(scores_neg) if len(scores_neg) > 0 else np.array([0])
    # unify grid
    grid = np.linspace(0, 1, 500)
    from bisect import bisect_right
    def ecdf(scores, x):
        import math
        if len(scores) == 0:
            return 0.0
        return np.searchsorted(scores, x, side='right') / len(scores)
    cdf_p = np.array([ecdf(scores_pos, g) for g in grid])
    cdf_n = np.array([ecdf(scores_neg, g) for g in grid])
    ks = np.max(np.abs(cdf_p - cdf_n)) if len(scores_pos) and len(scores_neg) else 0.0
    plt.figure(figsize=(7, 5))
    plt.plot(grid, cdf_p, color=BLUE, label="Pos CDF")
    plt.plot(grid, cdf_n, color=BLUE_LIGHT, label="Neg CDF")
    plt.title(f"KS Curve (KS={ks:.3f})")
    plt.xlabel("Score")
    plt.ylabel("Cumulative probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_calibration(y_true, y_proba, out_path):
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


def plot_probability_distributions(y_true, y_proba, out_path):
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


def plot_threshold_sweep(y_true, y_proba, out_path):
    thresholds = np.linspace(0.0, 1.0, 201)
    accs = []
    f1s = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        accs.append(accuracy_score(y_true, preds))
        f1s.append(f1_score(y_true, preds, zero_division=0))
    best_acc_idx = int(np.argmax(accs))
    best_f1_idx = int(np.argmax(f1s))
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, accs, color=BLUE, lw=2, label="Accuracy")
    plt.plot(thresholds, f1s, color=BLUE_DARK, lw=2, linestyle=":", label="F1")
    plt.axvline(thresholds[best_acc_idx], color=BLUE_LIGHT, linestyle="--", label=f"Best Acc t={thresholds[best_acc_idx]:.3f}")
    plt.axvline(thresholds[best_f1_idx], color="#9ecae1", linestyle="--", label=f"Best F1 t={thresholds[best_f1_idx]:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep (Accuracy & F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_feature_importance_gain(booster: xgb.Booster, feature_names: List[str], out_path: str, top_k: int = 25):
    gain_importance = booster.get_score(importance_type="gain")
    if not gain_importance:
        print("No feature importance available from model.")
        return
    # Ensure all features included
    for name in feature_names:
        gain_importance.setdefault(name, 0.0)
    items = sorted(gain_importance.items(), key=lambda kv: kv[1], reverse=True)
    top_items = items[:top_k]
    names = [k for k, _ in top_items][::-1]
    vals = [v for _, v in top_items][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(names, vals, color=BLUE)
    plt.xlabel("Gain (importance)")
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
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

def run_lgbm_cv(X: pd.DataFrame, y: pd.Series, recall_target: float = 0.85, n_splits: int = 5) -> dict:
    if not HAS_LGB:
        return {"error": "lightgbm_not_installed"}
    X_vals = X.values
    y_vals = y.values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros_like(y_vals, dtype=float)
    for fold, (tr, va) in enumerate(skf.split(X_vals, y_vals), start=1):
        X_tr, y_tr = X_vals[tr], y_vals[tr]
        X_va, y_va = X_vals[va], y_vals[va]
        # upsample attack on training fold
        X_tr, y_tr = upsample_minority(X_tr, y_tr, random_state=42 + fold)
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42 + fold,
        )
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=["auc"],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_proba[va] = clf.predict_proba(X_va)[:, 1]

    # thresholds on OOF
    thresholds = np.linspace(0.0, 1.0, 1001)
    preds = np.array([(oof_proba >= t).astype(int) for t in thresholds])
    accs = np.array([accuracy_score(y_vals, p) for p in preds])
    f1s = np.array([f1_score(y_vals, p, zero_division=0) for p in preds])
    recs = np.array([recall_score(y_vals, p, zero_division=0) for p in preds])

    t_acc = float(thresholds[int(np.argmax(accs))])
    t_f1 = float(thresholds[int(np.argmax(f1s))])
    valid_idx = np.where(recs >= recall_target)[0]
    t_rec = float(thresholds[int(valid_idx[0])]) if len(valid_idx) > 0 else float(thresholds[int(np.argmax(recs))])

    # metrics per op-point
    def metrics_at(t):
        yp = (oof_proba >= t).astype(int)
        return dict(
            acc=float(accuracy_score(y_vals, yp)),
            prec=float(precision_score(y_vals, yp, zero_division=0)),
            rec=float(recall_score(y_vals, yp, zero_division=0)),
            f1=float(f1_score(y_vals, yp, zero_division=0)),
        )

    fpr, tpr, _ = roc_curve(y_vals, oof_proba)
    roc_auc = float(auc(fpr, tpr))
    pr_prec, pr_rec, _ = precision_recall_curve(y_vals, oof_proba)
    pr_auc = float(np.trapz(pr_prec[::-1], pr_rec[::-1]))

    summary = {
        "thresholds": {"acc_opt": t_acc, "f1_opt": t_f1, "rec_target": t_rec},
        "metrics_acc_opt": metrics_at(t_acc),
        "metrics_f1_opt": metrics_at(t_f1),
        "metrics_rec_target": metrics_at(t_rec),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cv_splits": n_splits,
    }
    # save summary
    out_path = os.path.join(os.path.dirname(__file__), "enh_synth_cv_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")
    return summary


def main():
    print("Loading and preprocessing enhanced synthetic dataset...")
    X, y, feature_names = load_and_preprocess(DATA_PATH)
    print(f"Data shape: X={X.shape}, y positives={(y==1).sum()}, negatives={(y==0).sum()}")

    # CV with LightGBM if available (for paper-quality summary)
    if HAS_LGB:
        print("Running stratified 5-fold CV with LightGBM (upsampled Attack)...")
        cv_summary = run_lgbm_cv(X, y, recall_target=0.85, n_splits=5)
        if "error" not in cv_summary:
            print("CV summary (LightGBM):")
            print(cv_summary)
        else:
            print("LightGBM not installed; skipping CV.")

    print("Training XGBoost model with early stopping (single split) and generating plots...")
    recall_target = 0.85
    booster, (X_val, y_val, y_proba, y_pred_f1, y_pred_acc, y_pred_rec), metrics = train_xgb(X, y, feature_names, target_recall=recall_target)
    print("Validation report (F1-optimal, balanced):")
    print(classification_report(y_val, y_pred_f1, target_names=["Benign", "Attack"], digits=4))
    print("Validation report (Accuracy-optimal):")
    print(classification_report(y_val, y_pred_acc, target_names=["Benign", "Attack"], digits=4))
    print(f"Validation report (Recall-targeted, target={recall_target:.2f}):")
    print(classification_report(y_val, y_pred_rec, target_names=["Benign", "Attack"], digits=4))

    # Plots
    base_dir = os.path.dirname(__file__)
    # Common curves
    plot_roc(y_val, y_proba, os.path.join(base_dir, "enh_roc.png"))
    plot_pr(y_val, y_proba, os.path.join(base_dir, "enh_pr.png"))
    plot_det(y_val, y_proba, os.path.join(base_dir, "enh_det.png"))
    plot_ks(y_val, y_proba, os.path.join(base_dir, "enh_ks.png"))
    plot_calibration(y_val, y_proba, os.path.join(base_dir, "enh_calibration.png"))
    plot_probability_distributions(y_val, y_proba, os.path.join(base_dir, "enh_probability_distributions.png"))
    plot_threshold_sweep(y_val, y_proba, os.path.join(base_dir, "enh_threshold_sweep.png"))
    plot_feature_importance_gain(booster, feature_names, os.path.join(base_dir, "enh_feature_importance_gain.png"))

    # Confusion matrices per operating point
    plot_confusion_matrices(y_val, y_pred_f1,
                            os.path.join(base_dir, "enh_confusion_matrix_f1.png"),
                            os.path.join(base_dir, "enh_confusion_matrix_f1_normalized.png"))
    plot_confusion_matrices(y_val, y_pred_acc,
                            os.path.join(base_dir, "enh_confusion_matrix_accopt.png"),
                            os.path.join(base_dir, "enh_confusion_matrix_accopt_normalized.png"))
    plot_confusion_matrices(y_val, y_pred_rec,
                            os.path.join(base_dir, "enh_confusion_matrix_recall.png"),
                            os.path.join(base_dir, "enh_confusion_matrix_recall_normalized.png"))

    # PNG tables of results
    save_metrics_table(y_val, y_pred_f1, os.path.join(base_dir, "enh_results_table_f1.png"), title="F1-optimal Results")
    save_metrics_table(y_val, y_pred_acc, os.path.join(base_dir, "enh_results_table_accopt.png"), title="Accuracy-optimal Results")
    save_metrics_table(y_val, y_pred_rec, os.path.join(base_dir, "enh_results_table_recall.png"), title=f"Recall-targeted Results (target={recall_target:.2f})")

    # Write markdown tables with per-class metrics for all operating points
    def report_table(y_true, y_pred) -> str:
        rep = classification_report(y_true, y_pred, target_names=["Benign", "Attack"], output_dict=True, zero_division=0)
        lines = []
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        for cls in ["Benign", "Attack"]:
            m = rep[cls]
            lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |")
        acc = rep["accuracy"] if "accuracy" in rep else accuracy_score(y_true, y_pred)
        lines.append("")
        lines.append(f"Accuracy: {acc:.4f}")
        return "\n".join(lines)

    tables_md = [
        "## Enhanced Synthetic Results Tables",
        "",
        "### F1-optimal operating point",
        report_table(y_val, y_pred_f1),
        "",
        "### Accuracy-optimal operating point",
        report_table(y_val, y_pred_acc),
        "",
        f"### Recall-targeted operating point (target={recall_target:.2f})",
        report_table(y_val, y_pred_rec),
    ]
    # Include CV summary if available
    cv_path = os.path.join(base_dir, "enh_synth_cv_summary.txt")
    if os.path.exists(cv_path):
        try:
            with open(cv_path, "r", encoding="utf-8") as f:
                cv_txt = f.read()
            tables_md.append("")
            tables_md.append("### LightGBM 5-fold CV summary (OOF)")
            tables_md.append("```")
            tables_md.append(cv_txt.strip())
            tables_md.append("```")
        except Exception:
            pass
    out_tables = os.path.join(base_dir, "ENHANCED_SYNTHETIC_RESULTS_TABLES.md")
    with open(out_tables, "w", encoding="utf-8") as f:
        f.write("\n".join(tables_md) + "\n")

    print("Saved model, thresholds, metrics, plots, PNG tables, and markdown tables in enhanced_synthetic_cyber_attack/ ")


if __name__ == "__main__":
    main()


