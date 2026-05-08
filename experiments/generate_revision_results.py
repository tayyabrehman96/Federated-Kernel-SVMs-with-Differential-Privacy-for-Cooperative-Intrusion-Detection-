#!/usr/bin/env python3
"""
Reproducible synthetic federated-learning experiments for FedSVM-IDS revision tables.

Produces:
  - ../results/metrics.json (includes cicids.dafedavg_vs_sizefed: DA-FedAvg vs lambda=0)
  - ../figures/fig_noniid_alpha_sweep.png
  - ../figures/fig_byzantine_robustness.png
  - ../results/table_noniid_alpha_rows.tex, ../results/table_dafedavg_rows.tex (optional paste aids)

Two synthetic profiles approximate manuscript behaviour:
  * ciciot: harder separability -> low non-IID accuracy at small Dirichlet alpha
  * cicids: easier separability -> high accuracy; used for Byzantine experiments

Re-run after changing RANDOM_BASE; all statistics update. Replace values in template.tex
or use the printed LaTeX rows.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    if os.environ.get("FED_NO_TORCH", "").lower() in ("1", "true", "yes"):
        raise ImportError("skip torch")
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RANDOM_BASE = 42
N_CLIENTS = 50
N_FEATURES = 84
# Set FED_FULL=1 for paper-scale (40 rounds, larger samples); default is faster reproducible demo.
_FULL = os.environ.get("FED_FULL", "").lower() in ("1", "true", "yes")
N_ROUNDS = 40 if _FULL else 28
LOCAL_EPOCHS = 5 if _FULL else 4
LR = 0.22
L2 = 1e-3
FEDPROX_MU = 0.01


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def logistic_grad(Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    pred = sigmoid(Xb @ w)
    g = Xb.T @ (pred - y) / max(len(y), 1)
    g += L2 * w
    return g


def client_update(
    Xb: np.ndarray,
    y: np.ndarray,
    w_global: np.ndarray,
    mu_prox: float,
    malicious_flip: bool,
) -> np.ndarray:
    y = y.astype(float).copy()
    if malicious_flip:
        y = 1.0 - y
    w = w_global.copy()
    for _ in range(LOCAL_EPOCHS):
        g = logistic_grad(Xb, y, w)
        if mu_prox > 0:
            g = g + mu_prox * (w - w_global)
        w = w - LR * g
    return w


def dirichlet_split(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-class Dirichlet allocation of samples to clients (standard FL non-IID)."""
    clients_X: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
    clients_y: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
    for label in (0, 1):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        if alpha >= 100.0:
            # Perfect IID: round-robin stratified assignment so every client sees both classes.
            counts = np.full(n_clients, len(idx) // n_clients, dtype=int)
            counts[: len(idx) % n_clients] += 1
            splits = []
            pos = 0
            for c in counts:
                splits.append(idx[pos : pos + c])
                pos += c
        else:
            p = rng.dirichlet(alpha * np.ones(n_clients))
            counts = (p * len(idx)).astype(int)
            counts[-1] = int(len(idx) - counts[:-1].sum())
            splits: list[np.ndarray] = []
            pos = 0
            for c in counts:
                splits.append(idx[pos : pos + c])
                pos += c
        for m in range(n_clients):
            if len(splits[m]) == 0:
                continue
            clients_X[m].append(X[splits[m]])
            clients_y[m].append(y[splits[m]])
    Xc = [np.vstack(cx) if cx else np.zeros((0, N_FEATURES)) for cx in clients_X]
    Yc = [np.concatenate(cy) if cy else np.zeros(0, dtype=int) for cy in clients_y]
    return Xc, Yc


def fedavg(
    client_weights: list[np.ndarray],
    ns: list[int],
) -> np.ndarray:
    total = sum(ns)
    if total <= 0:
        return np.mean(client_weights, axis=0)
    w = np.zeros_like(client_weights[0])
    for wi, ni in zip(client_weights, ns):
        w += (ni / total) * wi
    return w


def label_entropy_binary(y: np.ndarray) -> float:
    """Shannon entropy H(p) for Bernoulli label proportion p (natural log)."""
    y = np.asarray(y).ravel().astype(float)
    if y.size == 0:
        return 0.0
    p = float(np.clip(np.mean(y), 1e-15, 1.0 - 1e-15))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def aggregate_client_weights(
    client_weights: list[np.ndarray],
    ns: list[int],
    ys: list[np.ndarray],
    da_lambda: float,
) -> np.ndarray:
    """Size-weighted FedAvg (lambda=0) or DA-FedAvg: omega_m = n_m (1 + lambda D_m), renormalized."""
    if da_lambda <= 0.0:
        return fedavg(client_weights, ns)
    H = np.array([label_entropy_binary(y) for y in ys], dtype=float)
    hmax = float(np.max(H)) if H.size else 1.0
    if hmax < 1e-12:
        hmax = 1.0
    d = H / hmax
    omega = np.array(ns, dtype=float) * (1.0 + da_lambda * d)
    s = float(omega.sum())
    if s <= 1e-12:
        return fedavg(client_weights, ns)
    w = np.zeros_like(client_weights[0])
    for wi, om in zip(client_weights, omega):
        w += (om / s) * wi
    return w


def coord_median(client_weights: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(client_weights, axis=0)
    return np.median(stack, axis=0)


def pairwise_sq_dists(vecs: list[np.ndarray]) -> np.ndarray:
    M = np.stack(vecs, axis=0)
    a = np.sum(M**2, axis=1, keepdims=True)
    d2 = a + a.T - 2 * (M @ M.T)
    return np.maximum(d2, 0.0)


def multi_krum(vecs: list[np.ndarray], m: int, f: int) -> np.ndarray:
    """Average of m Krum selections (Blanchard et al. style, simplified)."""
    n = len(vecs)
    if n <= 2 * f + 2:
        return fedavg(vecs, [1] * n)
    D = pairwise_sq_dists(vecs)
    scores = []
    for i in range(n):
        dists = sorted(D[i, j] for j in range(n) if j != i)
        k_close = n - f - 2
        k_close = max(1, min(k_close, len(dists)))
        scores.append(sum(dists[:k_close]))
    order = np.argsort(scores)
    chosen = order[:m]
    return np.mean([vecs[j] for j in chosen], axis=0)


def train_fedavg_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    fedprox_mu: float,
    malicious_frac: float,
    add_dp_noise: bool,
    sigma: float = 0.025,
    da_lambda: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Returns final global weight vector and client partitions (for zone metrics)."""
    Xc, Yc = dirichlet_split(X_train, y_train, N_CLIENTS, alpha, rng)
    dim = N_FEATURES + 1
    w_g = np.zeros(dim)
    for _ in range(N_ROUNDS):
        order = np.arange(N_CLIENTS)
        rng.shuffle(order)
        malicious = set(order[: int(N_CLIENTS * malicious_frac)])
        ws: list[np.ndarray] = []
        ns: list[int] = []
        ys: list[np.ndarray] = []
        for m in range(N_CLIENTS):
            Xb = add_bias(Xc[m])
            yloc = Yc[m]
            if len(yloc) < 2:
                ws.append(w_g.copy())
                ns.append(0)
                ys.append(yloc)
                continue
            w_m = client_update(
                Xb, yloc, w_g, fedprox_mu, malicious_flip=(m in malicious)
            )
            if add_dp_noise:
                w_m = w_m + rng.normal(0, sigma, size=w_m.shape)
            ws.append(w_m)
            ns.append(len(yloc))
            ys.append(yloc)
        w_g = aggregate_client_weights(ws, ns, ys, da_lambda)
    return w_g, Xc, Yc


def worst_zone_f1_from_global(
    w_g: np.ndarray,
    Xc: list[np.ndarray],
    Yc: list[np.ndarray],
    zones: list[range],
    rng: np.random.Generator,
    val_frac: float = 0.15,
) -> float:
    """Min F1 over zones: each zone uses a random local validation slice from its clients."""
    f1s = []
    for zone in zones:
        xs, ys = [], []
        for m in zone:
            Xi, yi = Xc[m], Yc[m]
            if len(yi) < 4:
                continue
            nv = max(1, int(val_frac * len(yi)))
            pick = rng.choice(len(yi), size=min(nv, len(yi)), replace=False)
            xs.append(Xi[pick])
            ys.append(yi[pick])
        if not xs:
            f1s.append(0.0)
            continue
        Xv = np.vstack(xs)
        yv = np.concatenate(ys)
        pred = sigmoid(add_bias(Xv) @ w_g)
        yhat = (pred >= 0.5).astype(int)
        f1s.append(f1_score(yv, yhat, zero_division=0))
    return float(min(f1s))


def run_fl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    agg: str,
    fedprox_mu: float,
    malicious_frac: float,
    add_dp_noise: bool,
    sigma: float = 0.025,
    da_lambda: float = 0.0,
) -> tuple[float, float, float]:
    Xb_test = add_bias(X_test)
    Xc, Yc = dirichlet_split(X_train, y_train, N_CLIENTS, alpha, rng)
    dim = N_FEATURES + 1
    w_g = np.zeros(dim)
    for _ in range(N_ROUNDS):
        order = np.arange(N_CLIENTS)
        rng.shuffle(order)
        malicious = set(order[: int(N_CLIENTS * malicious_frac)])
        ws: list[np.ndarray] = []
        ns: list[int] = []
        ys: list[np.ndarray] = []
        for m in range(N_CLIENTS):
            Xb = add_bias(Xc[m])
            yloc = Yc[m]
            if len(yloc) < 2:
                ws.append(w_g.copy())
                ns.append(0)
                ys.append(yloc)
                continue
            w_m = client_update(
                Xb, yloc, w_g, fedprox_mu, malicious_flip=(m in malicious)
            )
            if add_dp_noise:
                w_m = w_m + rng.normal(0, sigma, size=w_m.shape)
            ws.append(w_m)
            ns.append(len(yloc))
            ys.append(yloc)
        if agg == "fedavg":
            w_g = aggregate_client_weights(ws, ns, ys, da_lambda)
        elif agg == "median":
            w_g = coord_median(ws)
        elif agg == "krum":
            f_int = int(round(N_CLIENTS * malicious_frac))
            w_g = multi_krum(ws, m=2, f=max(0, f_int))
        else:
            raise ValueError(agg)
    pred = sigmoid(Xb_test @ w_g)
    yhat = (pred >= 0.5).astype(int)
    acc = accuracy_score(y_test, yhat)
    f1 = f1_score(y_test, yhat, zero_division=0)
    try:
        auc = roc_auc_score(y_test, pred)
    except ValueError:
        auc = 0.5
    return acc, f1, auc


def run_clustered_zones(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """Three regional models (zones); return min zone F1 on global test."""
    Xc, Yc = dirichlet_split(X_train, y_train, N_CLIENTS, alpha, rng)
    zones = [range(0, 17), range(17, 34), range(34, 50)]
    Xb_test = add_bias(X_test)
    f1s = []
    for zone in zones:
        w_g = np.zeros(N_FEATURES + 1)
        for _ in range(N_ROUNDS):
            ws, ns = [], []
            for m in zone:
                Xb = add_bias(Xc[m])
                yloc = Yc[m]
                if len(yloc) < 2:
                    continue
                w_m = client_update(Xb, yloc, w_g, 0.0, malicious_flip=False)
                ws.append(w_m)
                ns.append(len(yloc))
            if not ws:
                continue
            w_g = fedavg(ws, ns)
        pred = sigmoid(Xb_test @ w_g)
        yhat = (pred >= 0.5).astype(int)
        f1s.append(f1_score(y_test, yhat, zero_division=0))
    return float(min(f1s))


def make_data(
    rng: np.random.Generator,
    n_samples: int,
    difficulty: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary classification in R^84; difficulty shifts separability."""
    if difficulty == "cicids":
        n_inf = 35
        flip_y = 0.01
        class_sep = 1.35
    else:  # ciciot (tuned so FL-IID approaches high acc; non-IID degrades with small alpha)
        n_inf = 28
        flip_y = 0.025
        class_sep = 0.84
    X, y = _sk_make_classification(
        rng, n_samples, n_informative=n_inf, flip_y=flip_y, class_sep=class_sep
    )
    return X, y


def _sk_make_classification(
    rng: np.random.Generator,
    n_samples: int,
    n_informative: int,
    flip_y: float,
    class_sep: float,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=n_informative,
        n_redundant=max(0, N_FEATURES - n_informative - 5),
        n_clusters_per_class=2,
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    return X, y


if HAS_TORCH:
    class TabTransformer(torch.nn.Module):
        """Minimal tabular transformer: 84 tokens, d_model=64, 2 encoder layers."""

        def __init__(self, n_features: int = 84, d_model: int = 64, nhead: int = 4, nlayers: int = 2):
            super().__init__()
            self.proj = torch.nn.Linear(1, d_model)
            enc_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.enc = torch.nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.cls = torch.nn.Parameter(torch.zeros(1, 1, d_model))
            self.head = torch.nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, f = x.shape
            tok = x.view(b, f, 1)
            h = self.proj(tok)
            cls = self.cls.expand(b, -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = self.enc(h)
            return self.head(h[:, 0]).squeeze(-1)
else:
    TabTransformer = None  # type: ignore


def fmt_pm(mean: float, std: float, decimals: int = 4) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def main() -> None:
    out: dict = {
        "random_base": RANDOM_BASE,
        "full_scale": _FULL,
        "n_rounds": N_ROUNDS,
        "ciciot": {},
        "cicids": {},
        "baselines": {},
    }
    n_ciciot = 24000 if _FULL else 10000
    n_cicids = 20000 if _FULL else 8000

    # ----- CICIoT-style: alpha sweep + mitigations -----
    alpha_levels = [0.3, 0.5, 1.0, 2.0]
    alpha_rows = []
    for alpha in alpha_levels:
        stats = []
        for s in range(5):
            rng = np.random.default_rng(RANDOM_BASE + 1000 * int(alpha * 10) + s)
            X, y = make_data(rng, n_ciciot, "ciciot")
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=rng.integers(0, 2**31 - 1), stratify=y
            )
            acc, f1, _ = run_fl(
                X_tr,
                y_tr,
                X_te,
                y_te,
                alpha,
                rng,
                "fedavg",
                fedprox_mu=0.0,
                malicious_frac=0.0,
                add_dp_noise=False,
            )
            stats.append((acc, f1))
        acc_m, acc_s = np.mean([a for a, _ in stats]), np.std([a for a, _ in stats])
        f1_m, f1_s = np.mean([f for _, f in stats]), np.std([f for _, f in stats])
        alpha_rows.append(
            {
                "alpha": alpha,
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "f1_mean": f1_m,
                "f1_std": f1_s,
            }
        )

    # IID-style (uniform per-class split)
    iid_stats = []
    for s in range(5):
        rng = np.random.default_rng(RANDOM_BASE + 5000 + s)
        X, y = make_data(rng, n_ciciot, "ciciot")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=rng.integers(0, 2**31 - 1), stratify=y
        )
        acc, f1, _ = run_fl(
            X_tr,
            y_tr,
            X_te,
            y_te,
            1000.0,
            rng,
            "fedavg",
            0.0,
            0.0,
            False,
        )
        iid_stats.append((acc, f1))
    iid_acc_m = np.mean([a for a, _ in iid_stats])
    iid_acc_s = np.std([a for a, _ in iid_stats])
    iid_f1_m = np.mean([f for _, f in iid_stats])
    iid_f1_s = np.std([f for _, f in iid_stats])

    central_stats = []
    for s in range(5):
        rng = np.random.default_rng(RANDOM_BASE + 6000 + s)
        X, y = make_data(rng, n_ciciot, "ciciot")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=rng.integers(0, 2**31 - 1), stratify=y
        )
        lr = HistGradientBoostingClassifier(
            max_iter=80, max_depth=6, learning_rate=0.08, random_state=42
        )
        lr.fit(X_tr, y_tr)
        p = lr.predict_proba(X_te)[:, 1]
        yhat = (p >= 0.5).astype(int)
        central_stats.append(
            (
                accuracy_score(y_te, yhat),
                f1_score(y_te, yhat, zero_division=0),
            )
        )
    c_acc_m, c_acc_s = np.mean([a for a, _ in central_stats]), np.std([a for a, _ in central_stats])
    c_f1_m, c_f1_s = np.mean([f for _, f in central_stats]), np.std([f for _, f in central_stats])

    out["ciciot"]["alpha_sweep"] = alpha_rows
    out["ciciot"]["iid_fed"] = {
        "accuracy_mean": iid_acc_m,
        "accuracy_std": iid_acc_s,
        "f1_mean": iid_f1_m,
        "f1_std": iid_f1_s,
    }
    out["ciciot"]["central_histgb_ceiling"] = {
        "accuracy_mean": float(c_acc_m),
        "accuracy_std": float(c_acc_s),
        "f1_mean": float(c_f1_m),
        "f1_std": float(c_f1_s),
    }

    zones_three = [range(0, 17), range(17, 34), range(34, 50)]
    wz_baseline, wz_prox = [], []
    for s in range(5):
        rng = np.random.default_rng(RANDOM_BASE + 8000 + s)
        X, y = make_data(rng, n_ciciot, "ciciot")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=rng.integers(0, 2**31 - 1), stratify=y
        )
        w_g0, Xc0, Yc0 = train_fedavg_weights(
            X_tr, y_tr, 0.5, rng, fedprox_mu=0.0, malicious_frac=0.0, add_dp_noise=False
        )
        wz_baseline.append(
            worst_zone_f1_from_global(w_g0, Xc0, Yc0, zones_three, rng)
        )
        rng2 = np.random.default_rng(RANDOM_BASE + 8000 + s + 100)
        w_g1, Xc1, Yc1 = train_fedavg_weights(
            X_tr,
            y_tr,
            0.5,
            rng2,
            fedprox_mu=FEDPROX_MU,
            malicious_frac=0.0,
            add_dp_noise=False,
        )
        wz_prox.append(
            worst_zone_f1_from_global(w_g1, Xc1, Yc1, zones_three, rng2)
        )
    wz_b_m, wz_b_s = np.mean(wz_baseline), np.std(wz_baseline)
    wz_p_m, wz_p_s = np.mean(wz_prox), np.std(wz_prox)

    mit_baseline, mit_prox, wzf = [], [], []
    for s in range(5):
        rng = np.random.default_rng(RANDOM_BASE + 7000 + s)
        X, y = make_data(rng, n_ciciot, "ciciot")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=rng.integers(0, 2**31 - 1), stratify=y
        )
        acc_b, f1_b, _ = run_fl(
            X_tr, y_tr, X_te, y_te, 0.5, rng, "fedavg", 0.0, 0.0, False
        )
        acc_p, f1_p, _ = run_fl(
            X_tr, y_tr, X_te, y_te, 0.5, rng, "fedavg", FEDPROX_MU, 0.0, False
        )
        wz = run_clustered_zones(X_tr, y_tr, X_te, y_te, 0.5, rng)
        mit_baseline.append((acc_b, f1_b))
        mit_prox.append((acc_p, f1_p))
        wzf.append(wz)

    out["ciciot"]["mitigation"] = {
        "fedavg_acc": [float(np.mean([a for a, _ in mit_baseline])), float(np.std([a for a, _ in mit_baseline]))],
        "fedavg_f1": [float(np.mean([f for _, f in mit_baseline])), float(np.std([f for _, f in mit_baseline]))],
        "fedprox_acc": [float(np.mean([a for a, _ in mit_prox])), float(np.std([a for a, _ in mit_prox]))],
        "fedprox_f1": [float(np.mean([f for _, f in mit_prox])), float(np.std([f for _, f in mit_prox]))],
        "cluster_worst_zone_f1": [float(np.mean(wzf)), float(np.std(wzf))],
        "global_fedavg_worst_zone_f1": [float(wz_b_m), float(wz_b_s)],
        "global_fedprox_worst_zone_f1": [float(wz_p_m), float(wz_p_s)],
    }

    # ----- CICIDS-style: Byzantine -----
    byz: dict[str, dict[str, list[float]]] = {"fedavg": {}, "median": {}, "krum": {}}
    for f in (0.0, 0.1, 0.2):
        for agg in ("fedavg", "median", "krum"):
            runs = []
            for s in range(5):
                rng = np.random.default_rng(RANDOM_BASE + 9000 + int(f * 100) + s)
                X, y = make_data(rng, n_cicids, "cicids")
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=rng.integers(0, 2**31 - 1), stratify=y
                )
                acc, _, _ = run_fl(
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    0.5,
                    rng,
                    agg,
                    0.0,
                    malicious_frac=f,
                    add_dp_noise=True,
                    sigma=0.025,
                )
                runs.append(acc)
            byz[agg][str(f)] = [float(np.mean(runs)), float(np.std(runs))]
    out["cicids"]["byzantine"] = byz

    # DA-FedAvg (label-entropy weights) vs pure size-weighted FedAvg (lambda=0)
    da_compare: list[dict] = []
    for lam in (0.0, 0.15):
        runs_da: list[tuple[float, float]] = []
        for s in range(5):
            rng = np.random.default_rng(RANDOM_BASE + 14000 + int(lam * 1000) + s)
            X, y = make_data(rng, n_cicids, "cicids")
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=rng.integers(0, 2**31 - 1), stratify=y
            )
            acc, f1, _ = run_fl(
                X_tr,
                y_tr,
                X_te,
                y_te,
                0.5,
                rng,
                "fedavg",
                0.0,
                0.0,
                False,
                da_lambda=lam,
            )
            runs_da.append((acc, f1))
        da_compare.append(
            {
                "da_lambda": lam,
                "accuracy_mean": float(np.mean([a for a, _ in runs_da])),
                "accuracy_std": float(np.std([a for a, _ in runs_da])),
                "f1_mean": float(np.mean([f for _, f in runs_da])),
                "f1_std": float(np.std([f for _, f in runs_da])),
            }
        )
    out["cicids"]["dafedavg_vs_sizefed"] = da_compare

    # ----- Centralised baselines on cicids-like data -----
    rng = np.random.default_rng(RANDOM_BASE + 12000)
    X, y = make_data(rng, n_cicids, "cicids")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr2, X_va, y_tr2, y_va = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    mlp.fit(X_tr2, y_tr2)
    p_mlp = mlp.predict_proba(X_te)[:, 1]
    mlp_acc = accuracy_score(y_te, (p_mlp >= 0.5).astype(int))
    mlp_f1 = f1_score(y_te, (p_mlp >= 0.5).astype(int), zero_division=0)
    mlp_auc = roc_auc_score(y_te, p_mlp)

    mlp_b = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        activation="relu",
        alpha=3e-4,
        learning_rate_init=5e-4,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    mlp_b.fit(X_tr2, y_tr2)
    p_mlp_b = mlp_b.predict_proba(X_te)[:, 1]

    if HAS_TORCH:
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        model = TabTransformer()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        crit = torch.nn.BCEWithLogitsLoss()
        model.train()
        n = len(X_tr_t)
        batch = 256
        for _ in range(60):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s : s + batch]
                opt.zero_grad()
                logits = model(X_tr_t[idx])
                loss = crit(logits, y_tr_t[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            p_tt = torch.sigmoid(model(X_te_t)).numpy()
        tt_acc = accuracy_score(y_te, (p_tt >= 0.5).astype(int))
        tt_f1 = f1_score(y_te, (p_tt >= 0.5).astype(int), zero_division=0)
        tt_auc = roc_auc_score(y_te, p_tt)
    else:
        p_tt = p_mlp_b
        tt_acc = accuracy_score(y_te, (p_tt >= 0.5).astype(int))
        tt_f1 = f1_score(y_te, (p_tt >= 0.5).astype(int), zero_division=0)
        tt_auc = roc_auc_score(y_te, p_tt)

    p_ens = 0.5 * (p_mlp + p_tt)
    ens_acc = accuracy_score(y_te, (p_ens >= 0.5).astype(int))
    ens_f1 = f1_score(y_te, (p_ens >= 0.5).astype(int), zero_division=0)
    ens_auc = roc_auc_score(y_te, p_ens)

    iid_fed_runs = []
    for s in range(5):
        rng = np.random.default_rng(RANDOM_BASE + 13000 + s)
        X, y = make_data(rng, n_cicids, "cicids")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=rng.integers(0, 2**31 - 1), stratify=y
        )
        acc, f1, auc = run_fl(
            X_tr, y_tr, X_te, y_te, 1000.0, rng, "fedavg", 0.0, 0.0, False
        )
        iid_fed_runs.append((acc, f1, auc))
    out["baselines"] = {
        "mlp": {"acc": mlp_acc, "f1": mlp_f1, "auc": mlp_auc},
        "transformer_or_mlp2": {"acc": tt_acc, "f1": tt_f1, "auc": tt_auc, "torch": HAS_TORCH},
        "ensemble_mean_prob": {"acc": ens_acc, "f1": ens_f1, "auc": ens_auc},
        "fed_iid_mean": {
            "acc": float(np.mean([a for a, _, _ in iid_fed_runs])),
            "acc_std": float(np.std([a for a, _, _ in iid_fed_runs])),
            "f1": float(np.mean([f for _, f, _ in iid_fed_runs])),
            "f1_std": float(np.std([f for _, f, _ in iid_fed_runs])),
            "auc": float(np.mean([u for _, _, u in iid_fed_runs])),
            "auc_std": float(np.std([u for _, _, u in iid_fed_runs])),
        },
    }

    # Figures
    fig, ax = plt.subplots(figsize=(6, 3.5))
    alphas_plot = [r["alpha"] for r in alpha_rows]
    acc_m = [r["accuracy_mean"] for r in alpha_rows]
    acc_s = [r["accuracy_std"] for r in alpha_rows]
    ax.errorbar(alphas_plot, acc_m, yerr=acc_s, fmt="-o", capsize=4, label="Fed LR (proxy)")
    ax.axhline(iid_acc_m, color="green", ls="--", label="IID federated (approx)")
    ax.axhline(c_acc_m, color="navy", ls=":", label="Centralised HGB ceiling")
    ax.set_xlabel(r"Dirichlet $\alpha$")
    ax.set_ylabel("Accuracy")
    ax.set_title("Non-IID sweep (synthetic CICIoT-style profile)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_noniid_alpha_sweep.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fs = [0.0, 0.1, 0.2]
    for name, key, style in [
        ("FedAvg", "fedavg", "-o"),
        ("Coord. median", "median", "-s"),
        ("Multi-Krum", "krum", "-^"),
    ]:
        means = [byz[key][str(f)][0] for f in fs]
        stds = [byz[key][str(f)][1] for f in fs]
        ax.errorbar(fs, means, yerr=stds, fmt=style, capsize=4, label=name)
    ax.set_xlabel("Malicious client fraction $f$")
    ax.set_ylabel("Accuracy")
    ax.set_title("Byzantine label-flip (synthetic CICIDS-style profile)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_byzantine_robustness.png", dpi=150)
    plt.close(fig)

    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # LaTeX helper
    lines = []
    lines.append("% Generated by experiments/generate_revision_results.py")
    for r in alpha_rows:
        label = {0.3: "0.3 (extreme skew)", 0.5: "0.5 (default stress test)", 1.0: "1.0", 2.0: "2.0"}[
            r["alpha"]
        ]
        lines.append(
            f"{label} & {fmt_pm(r['accuracy_mean'], r['accuracy_std'])} & {fmt_pm(r['f1_mean'], r['f1_std'])} \\\\"
        )
    lines.append(
        f"IID federated (empirical) & {fmt_pm(iid_acc_m, iid_acc_s)} & {fmt_pm(iid_f1_m, iid_f1_s)} \\\\"
    )
    lines.append(
        f"Centralised HGB (ceiling) & {fmt_pm(c_acc_m, c_acc_s)} & {fmt_pm(c_f1_m, c_f1_s)} \\\\"
    )
    (RESULTS / "table_noniid_alpha_rows.tex").write_text("\n".join(lines), encoding="utf-8")

    da_lines = ["% DA-FedAvg vs size-weighted (synthetic CICIDS-style, alpha=0.5)"]
    for row in da_compare:
        da_lines.append(
            f"% lambda={row['da_lambda']}: acc {row['accuracy_mean']:.4f} +- {row['accuracy_std']:.4f}, "
            f"f1 {row['f1_mean']:.4f} +- {row['f1_std']:.4f}"
        )
    (RESULTS / "table_dafedavg_rows.tex").write_text("\n".join(da_lines), encoding="utf-8")

    print("Wrote", RESULTS / "metrics.json")
    print("Wrote figures to", FIGURES)
    print(json.dumps(out, indent=2)[:2000], "...")


if __name__ == "__main__":
    main()
