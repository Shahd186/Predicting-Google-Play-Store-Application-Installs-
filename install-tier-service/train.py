"""Train the final Random Forest on googleplaystore_classification.csv and publish it as rf-v1.

    python train.py --data googleplaystore_classification.csv [--raw Google-Playstore.csv]

Training uses the processed CSV as it is, exactly like Milestone3_with_MLP_imbalance.ipynb:
the 60 feature columns as float32, Installs_Class as the label, a stratified 80/20 split with
seed 42, and Size_MB standardised with the training split's mean/std.

--raw (recommended): the raw Kaggle CSV. Gives the exact Phase 2 statistics and the developer
frequency table the server needs to turn a user's input into the same 60 features, and checks
row by row that the server rebuilds the processed CSV (prints PASSED / FAILED).
Without it the statistics are recovered from the processed CSV and every developer is unseen.

Writes registry/ (the model) and data/train_X.npy, train_y.npy, test_X.npy, test_y.npy (used by retraining).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import common as C

RAW_COLUMNS = ["Category", "Free", "Ad Supported", "Privacy Policy", "Content Rating", "Size",
               "Minimum Android", "Developer Id", "Released", "Last Updated", "Scraped Time", "Maximum Installs"]


def load_processed(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in header if c not in C.NON_FEATURE_COLUMNS]
    needed = [C.F_FREE, C.F_ADS, C.F_SIZE, *C.SCALED_COLUMNS, C.F_EVERYONE, C.F_PRIVACY, *C.SIZE_TIER_COLUMNS]
    if C.LABEL_COLUMN not in header or any(c not in cols for c in needed):
        raise ValueError("This is not googleplaystore_classification.csv (columns are missing).")
    df = pd.read_csv(path, usecols=cols + [C.LABEL_COLUMN],
                     dtype={**dict.fromkeys(cols, np.float32), C.LABEL_COLUMN: str})
    y = df[C.LABEL_COLUMN].map({t: i for i, t in enumerate(C.TIERS)})
    if y.isna().any():
        raise ValueError(f"Unknown labels: {df.loc[y.isna(), C.LABEL_COLUMN].unique()[:5]}")
    exact = pd.read_csv(path, usecols=[C.F_SIZE, *C.SCALED_COLUMNS], dtype=np.float64)  # full precision
    return cols, df[cols].to_numpy(np.float32), y.to_numpy(np.int8), exact


def parity_check(stats, cols, raw_path, processed_path, n=50_000, tol=1e-4) -> bool:
    """Server features (raw rows -> FeaturePipeline) vs the same rows of the processed CSV."""
    raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS, nrows=n, low_memory=False).dropna(subset=["Maximum Installs"])
    pipe = C.FeaturePipeline(feature_names=cols, medians=stats["medians"], scaler=stats["scaler"],
                             developer_freq=stats["developer_freq"])  # size_mean=0, std=1: raw Size_MB as in the CSV
    ours = pipe.transform(C.canonical_from_dataset(raw)).astype(float)
    theirs = pd.read_csv(processed_path, usecols=cols, nrows=len(raw))[cols].to_numpy(float)
    diff = pd.Series(np.abs(ours - theirs).max(axis=0), index=cols).sort_values(ascending=False)
    print(f"Parity on the first {len(raw):,} apps, largest differences:\n{diff.head(5).to_string()}")
    ok = bool((diff <= tol).all())
    print("PASSED: the server builds the same features as Phase 2." if ok else
          f"FAILED: {list(diff[diff > tol].index)} differ from the processed CSV.")
    return ok


DEFAULT_CANDIDATES = "15x10,20x15,30x20,40x30"  # trees x min_samples_leaf


def model_size_gb(model) -> float:
    """Size of the model file as the server saves it (uncompressed); ~ its size in RAM."""
    fd, tmp = tempfile.mkstemp(suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(model, tmp)
        return os.path.getsize(tmp) / 1e9
    finally:
        os.remove(tmp)


def tier_line(m: dict) -> str:
    return "  ".join(f"{t} R={m['per_tier_recall'][t]:.3f}" for t in C.TIERS)


def search(X_train, y_train, candidates: str, max_gb: float) -> tuple[dict, list]:
    """Pick (trees, min_samples_leaf) on a validation split carved from the TRAINING split
    (the test split is not touched). Rule: highest validation macro-F1 among the candidates whose
    model fits max_gb; if none fits, the smallest model."""
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=0.1, stratify=y_train,
                                        random_state=C.SEED)
    X_fit, y_fit, X_val, y_val = X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx]
    rows = []
    print(f"\nModel search on a validation split ({len(y_val):,} apps), budget {max_gb} GB per model file:")
    for cand in candidates.split(","):
        n, leaf = (int(v) for v in cand.lower().split("x"))
        t0 = time.time()
        m = RandomForestClassifier(**dict(C.RF_PARAMS, n_estimators=n, min_samples_leaf=leaf)).fit(X_fit, y_fit)
        ev = C.evaluate(y_val, m.predict(X_val))
        row = {"n_estimators": n, "min_samples_leaf": leaf, "size_gb": round(model_size_gb(m), 3),
               "val_macro_f1": round(ev["macro_f1"], 4), "val_accuracy": round(ev["accuracy"], 4),
               "val_recall": {t: round(v, 3) for t, v in ev["per_tier_recall"].items()},
               "fit_seconds": round(time.time() - t0)}
        rows.append(row)
        print(f"  {n:3d} trees, leaf {leaf:3d}: macro-F1 {row['val_macro_f1']:.4f}, size {row['size_gb']:.2f} GB, "
              f"{row['fit_seconds']}s | {tier_line(ev)}", flush=True)
        del m
        gc.collect()
    del X_fit, X_val
    fits = [r for r in rows if r["size_gb"] <= max_gb]
    best = max(fits, key=lambda r: r["val_macro_f1"]) if fits else min(rows, key=lambda r: r["size_gb"])
    print(f"Chosen: {best['n_estimators']} trees, min_samples_leaf {best['min_samples_leaf']} "
          f"(validation macro-F1 {best['val_macro_f1']:.4f}, {best['size_gb']:.2f} GB)"
          + ("" if fits else "  -- no candidate fits the budget; picked the smallest"))
    return {"n_estimators": best["n_estimators"], "min_samples_leaf": best["min_samples_leaf"]}, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="googleplaystore_classification.csv")
    ap.add_argument("--raw", help="raw Kaggle Google-Playstore.csv (recommended)")
    ap.add_argument("--registry", default="registry")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-estimators", type=int, help="fewer trees for quick tests")
    ap.add_argument("--min-samples-leaf", type=int, help="larger = smaller model file (report default: 5)")
    ap.add_argument("--parity-rows", type=int, default=50_000)
    ap.add_argument("--search", nargs="?", const=DEFAULT_CANDIDATES, metavar="NxL,...",
                    help=f"try several trees x min_samples_leaf (default {DEFAULT_CANDIDATES}) and keep the best that fits --max-model-gb")
    ap.add_argument("--max-model-gb", type=float, default=0.4, help="model size budget for --search (6 GB laptop: 0.4)")
    args = ap.parse_args(argv)

    t0 = time.time()
    cols, X, y, exact = load_processed(args.data)
    print(f"Loaded {len(y):,} apps x {len(cols)} features in {time.time() - t0:.0f}s")

    if args.raw:
        stats = C.compute_phase2_stats(pd.read_csv(args.raw, usecols=RAW_COLUMNS, low_memory=False))
        print(f"Phase 2 statistics from the raw CSV: {len(stats['developer_freq']):,} developers")
        if not parity_check(stats, cols, args.raw, args.data, args.parity_rows):
            sys.exit("Stopping: fix the feature code before training.")
    else:
        stats = C.recover_from_processed(exact[C.SCALED_COLUMNS], exact[C.F_SIZE].to_numpy(), len(exact))
        print("No --raw: Phase 2 statistics recovered from the processed CSV; every developer counts as unseen.")

    # --- exactly as in the Milestone 3 notebook
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=C.SEED)
    j = cols.index(C.F_SIZE)
    size_mean, size_std = X[tr, j].mean(), X[tr, j].std()
    X[:, j] = (X[:, j] - size_mean) / size_std
    X_train, y_train, X_test, y_test = X[tr], y[tr], X[te], y[te]
    del X

    raw_vals = {c: exact[c].to_numpy() * stats["scaler"][c][1] + stats["scaler"][c][0]
                for c in (C.F_ANDROID, C.F_AGE, C.F_RECENCY)}
    ranges = {c: [float(v.min()), float(v.max())] for c, v in raw_vals.items()}
    ranges[C.F_SIZE] = [float(exact[C.F_SIZE].min()), float(exact[C.F_SIZE].max())]
    pipe = C.FeaturePipeline(feature_names=cols, medians=stats["medians"], scaler=stats["scaler"],
                             size_mean=float(size_mean), size_std=float(size_std),
                             developer_freq=stats["developer_freq"], ranges=ranges,
                             android_versions=sorted(set(np.round(raw_vals[C.F_ANDROID], 1).tolist())),
                             stats_source=stats["source"])

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    for name, arr in {"train_X": X_train, "train_y": y_train, "test_X": X_test, "test_y": y_test}.items():
        np.save(Path(args.data_dir) / f"{name}.npy", arr)  # plain .npy: retraining memory-maps them

    search_rows = None
    if args.search:
        chosen, search_rows = search(X_train, y_train, args.search, args.max_model_gb)
        params = dict(C.RF_PARAMS, **chosen)
    else:
        params = dict(C.RF_PARAMS, **({"n_estimators": args.n_estimators} if args.n_estimators else {}),
                      **({"min_samples_leaf": args.min_samples_leaf} if args.min_samples_leaf else {}))
    t_fit = time.time()
    model = RandomForestClassifier(**params).fit(X_train, y_train)
    print(f"\nFinal model: {params['n_estimators']} trees, min_samples_leaf {params['min_samples_leaf']}; "
          f"split {len(y_train):,} / {len(y_test):,}; fitted in {time.time() - t_fit:.0f}s")
    metrics = C.evaluate(y_test, model.predict(X_test))
    print(f"Test macro-F1 {metrics['macro_f1']:.4f} (report setting 100 trees x leaf 5: 0.4929), "
          f"accuracy {metrics['accuracy']:.4f}, within one tier {metrics['within_one_tier']:.3f}")
    print(f"Test recall:    {tier_line(metrics)}")
    print("Test precision: " + "  ".join(f"{t} P={metrics['per_tier_precision'][t]:.3f}" for t in C.TIERS))

    reg = C.Registry(args.registry)
    version = reg.next_version()
    path = reg.save({"version": version, "model": model, "pipeline": pipe, "rf_params": params,
                     "train_size": int(len(y_train)), "n_feedback": 0, "feedback_weight": None,
                     "metrics": metrics, "trained_at": datetime.now(timezone.utc).isoformat(),
                     "data_file": Path(args.data).name, "model_search": search_rows})
    reg.publish(version)
    reg.log({"event": "initial_training", "version": version, "published": True,
             "test_macro_f1": metrics["macro_f1"], "phase2_stats": stats["source"],
             "n_estimators": params["n_estimators"], "min_samples_leaf": params["min_samples_leaf"],
             "model_search": search_rows})
    nodes = sum(t.tree_.node_count for t in model.estimators_)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB, {nodes:,} tree nodes) and published {version}")


if __name__ == "__main__":
    main()
