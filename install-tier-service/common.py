"""Shared code for training, serving and retraining.

Sections: 1 settings · 2 Phase 2 features · 3 Phase 2 statistics · 4 metrics · 5 model registry
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# =============================================================== 1. settings
CODE_VERSION = "2026-11-27 · 60 features, with developer frequency"
LABEL_COLUMN = "Installs_Class"
NON_FEATURE_COLUMNS = ["Maximum Installs", "Minimum Installs", "Installs", "Rating Count",
                       "Log_Max_Installs", LABEL_COLUMN]  # dropped by the Milestone 3 notebook
TIERS = ["Low", "Moderate", "High", "Very High"]
TIER_UPPER_BOUNDS = [10_000, 100_000, 1_000_000]  # inclusive: 10,000 -> Low
TIER_RANGES = {"Low": [0, 10_000], "Moderate": [10_001, 100_000],
               "High": [100_001, 1_000_000], "Very High": [1_000_001, None]}

# Phase 2 column names, exactly as in googleplaystore_classification.csv
F_FREE, F_ADS, F_SIZE = "Free", "Ad Supported", "Size_MB"
F_ANDROID, F_DEV = "Minimum_Android_Numeric", "Developer_Frequency"
F_AGE, F_RECENCY = "App_Age_Months", "Months_Since_Last_Update"
F_EVERYONE, F_PRIVACY = "Is_Everyone", "Has_Privacy_Policy"
SCALED_COLUMNS = [F_ANDROID, F_DEV, F_AGE, F_RECENCY]  # StandardScaler fitted on ALL apps in Phase 2
CATEGORY_PREFIX = "Category_"
SIZE_TIER_COLUMNS = ["Size_Tiny", "Size_Small", "Size_Medium", "Size_Large", "Size_Very_Large"]
SIZE_TIER_EDGES = [10, 30, 100, 300]  # right=False: 10 MB -> Small
CATEGORY_MERGES = {"Educational": "Education", "Music": "Music & Audio"}
MISSING_DEVELOPER = "__MISSING_DEVELOPER__"
DAYS_PER_MONTH = 30.44
CONTENT_RATINGS = ["Everyone", "Everyone 10+", "Teen", "Mature 17+", "Adults only 18+", "Unrated"]
# Printed by the Phase 2 notebook; used only when the raw Kaggle CSV is not available.
PHASE2_PRINTED_SCALER = {F_ANDROID: (4.346075, 0.852839), F_DEV: (0.000048, 0.000211),
                         F_AGE: (33.347405, 25.049749), F_RECENCY: (18.345720, 18.237865)}

SEED = 42
RF_PARAMS = dict(n_estimators=100, criterion="gini", class_weight="balanced_subsample",
                 max_features=0.33, max_depth=None, min_samples_leaf=5, n_jobs=-1, random_state=SEED)

FEEDBACK_WEIGHTS = [1, 10, 100]
RECENT_HOLDOUT_FRACTION = 0.2
MIN_FEEDBACK_FOR_WEIGHT_SEARCH = 50  # below this: w = 10, gate on the fixed test set only
DEFAULT_FEEDBACK_WEIGHT = 10
RETRAIN_SAMPLE_TRIGGER = 1000
BOOTSTRAP_RESAMPLES = 1000
WHAT_IF_NOTABLE = 0.01


# =============================================================== 2. Phase 2 features
# Parsing functions copied from google_playstore_phase2_preprocessing.ipynb.
def size_to_mb(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(",", "").lower()
    if "varies" in s or s in {"", "nan", "none"}:
        return np.nan
    m = re.search(r"([0-9]*\.?[0-9]+)\s*([kmg]?)", s)
    if not m:
        return np.nan
    number, unit = float(m.group(1)), m.group(2)
    return number / 1024 if unit == "k" else number * 1024 if unit == "g" else number


def parse_android_version(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if "varies" in s or s in {"", "nan", "none"}:
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def to_binary(series: pd.Series) -> pd.Series:
    mapping = {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0,
               "TRUE": 1, "FALSE": 0, 1: 1, 0: 0, "1": 1, "0": 0}
    return series.map(mapping).fillna(0).astype("int8")


def merge_category(cat):
    if cat is None or (isinstance(cat, float) and np.isnan(cat)):
        return "Missing"
    return CATEGORY_MERGES.get(str(cat), str(cat))


def installs_to_tier(installs) -> np.ndarray:
    return np.searchsorted(np.asarray(TIER_UPPER_BOUNDS), np.asarray(installs), side="left")


def size_tier_index(size_mb) -> np.ndarray:
    return np.searchsorted(np.asarray(SIZE_TIER_EDGES), np.asarray(size_mb), side="right")


CANONICAL_COLUMNS = ["category", "free", "ad_supported", "privacy_policy", "content_rating", "size_mb",
                     "minimum_android", "developer", "released", "last_updated", "observation_date"]


def canonical_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Raw Kaggle rows -> canonical columns (Phase 2 semantics)."""
    out = pd.DataFrame(index=df.index)
    out["category"] = df["Category"].map(merge_category)
    out["free"] = to_binary(df["Free"])
    out["ad_supported"] = to_binary(df["Ad Supported"])
    out["privacy_policy"] = df["Privacy Policy"].notna().astype("int8")
    out["content_rating"] = df["Content Rating"].fillna("").astype(str)
    out["size_mb"] = df["Size"].map(size_to_mb).astype(float)
    out["minimum_android"] = df["Minimum Android"].map(parse_android_version).astype(float)
    out["developer"] = df["Developer Id"].fillna(MISSING_DEVELOPER).astype(str)
    out["released"] = pd.to_datetime(df["Released"], errors="coerce")
    out["last_updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")
    out["observation_date"] = pd.to_datetime(df["Scraped Time"], errors="coerce")
    return out


def canonical_from_request(req: dict) -> pd.DataFrame:
    """One validated API request -> one canonical row."""
    android = req.get("minimum_android")
    row = {
        "category": merge_category(req["category"]),
        "free": int(bool(req["free"])), "ad_supported": int(bool(req["ad_supported"])),
        "privacy_policy": int(bool(req["privacy_policy"])), "content_rating": req["content_rating"],
        "size_mb": np.nan if req.get("size_mb") is None else float(req["size_mb"]),
        "minimum_android": np.nan if android in (None, "", "Varies") else parse_android_version(android),
        "developer": (req.get("developer") or "").strip() or MISSING_DEVELOPER,
        "released": pd.Timestamp(req["released"]) if req.get("released") else pd.NaT,
        "last_updated": pd.Timestamp(req["last_updated"]),
        "observation_date": pd.Timestamp(req["observation_date"]),
    }
    return pd.DataFrame([row], columns=CANONICAL_COLUMNS)


def raw_numeric(canon: pd.DataFrame) -> dict:
    """Un-imputed numeric values; NaN where Phase 2 imputes the median."""
    sec = 60 * 60 * 24 * DAYS_PER_MONTH
    age = (canon["observation_date"] - canon["released"]).dt.total_seconds() / sec
    recency = ((canon["observation_date"] - canon["last_updated"]).dt.total_seconds() / sec).abs()
    return {F_SIZE: canon["size_mb"].astype(float), F_ANDROID: canon["minimum_android"].astype(float),
            F_AGE: age.where(age >= 0).astype(float), F_RECENCY: recency.astype(float)}


@dataclass
class FeaturePipeline:
    """Phase 2 statistics + feature order; turns canonical rows into the 60 model features."""
    feature_names: list = field(default_factory=list)
    medians: dict = field(default_factory=dict)
    scaler: dict = field(default_factory=dict)        # column -> (mean, scale) from Phase 2
    size_mean: float = 0.0                            # training-split Size_MB mean (Milestone 3)
    size_std: float = 1.0
    developer_freq: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)
    android_versions: list = field(default_factory=list)
    content_ratings: list = field(default_factory=lambda: list(CONTENT_RATINGS))
    stats_source: str = ""

    @property
    def categories(self) -> list:
        return [f[len(CATEGORY_PREFIX):] for f in self.feature_names if f.startswith(CATEGORY_PREFIX)]

    @property
    def has_developer_table(self) -> bool:
        return bool(self.developer_freq)

    def index_of(self, name: str) -> int:
        return self.feature_names.index(name)

    def _z(self, col, values):
        mean, scale = self.scaler[col]
        return (values - mean) / scale

    def transform(self, canon: pd.DataFrame) -> np.ndarray:
        n, raw = len(canon), raw_numeric(canon)
        filled = {c: raw[c].fillna(self.medians[c]).to_numpy() for c in raw}
        dev = canon["developer"].map(self.developer_freq).fillna(0.0).to_numpy(dtype=float)
        cols = {
            F_FREE: canon["free"].to_numpy(), F_ADS: canon["ad_supported"].to_numpy(),
            F_SIZE: (filled[F_SIZE].astype(np.float32) - np.float32(self.size_mean)) / np.float32(self.size_std),
            F_ANDROID: self._z(F_ANDROID, filled[F_ANDROID]), F_DEV: self._z(F_DEV, dev),
            F_AGE: self._z(F_AGE, filled[F_AGE]), F_RECENCY: self._z(F_RECENCY, filled[F_RECENCY]),
            F_EVERYONE: canon["content_rating"].fillna("").str.strip().str.lower().eq("everyone").to_numpy(),
            F_PRIVACY: canon["privacy_policy"].to_numpy(),
        }
        X = np.zeros((n, len(self.feature_names)), dtype=np.float32)
        index = {f: i for i, f in enumerate(self.feature_names)}
        for name, values in cols.items():
            X[:, index[name]] = values
        cat_idx = canon["category"].map(lambda c: index.get(CATEGORY_PREFIX + str(c)))
        rows = np.flatnonzero(cat_idx.notna().to_numpy())
        X[rows, cat_idx.iloc[rows].astype(int).to_numpy()] = 1.0
        tier_cols = np.array([index[c] for c in SIZE_TIER_COLUMNS])
        X[np.arange(n), tier_cols[size_tier_index(filled[F_SIZE])]] = 1.0
        return X

    def warnings_for(self, canon_row: pd.DataFrame) -> list[str]:
        out, row = [], canon_row.iloc[0]
        labels = {F_SIZE: "App size", F_ANDROID: "Minimum Android version",
                  F_AGE: "App age", F_RECENCY: "Time since last update"}
        for key, s in raw_numeric(canon_row).items():
            val = s.iloc[0]
            if pd.notna(val) and key in self.ranges and not self.ranges[key][0] <= val <= self.ranges[key][1]:
                lo, hi = self.ranges[key]
                out.append(f"{labels[key]} is outside the range seen in training ({lo:.1f} to {hi:.1f}); "
                           "the prediction is an extrapolation.")
        if pd.notna(row["released"]) and row["released"] > row["observation_date"]:
            out.append("The release date is after the observation date: the app is not released yet, "
                       "so its age is set to the training median.")
        if row["developer"] != MISSING_DEVELOPER:
            if not self.has_developer_table:
                out.append("Developer frequencies are not loaded on this server; the developer is treated as unseen.")
            elif row["developer"] not in self.developer_freq:
                out.append("This developer was not seen in training; developer frequency is set to 0.")
        return out


# =============================================================== 3. Phase 2 statistics
def compute_phase2_stats(raw_df: pd.DataFrame) -> dict:
    """Exact Phase 2 statistics from the raw Kaggle CSV (all apps), incl. the developer table."""
    canon = canonical_from_dataset(raw_df.dropna(subset=["Maximum Installs"]))
    raw = raw_numeric(canon)
    medians = {c: float(raw[c].median()) for c in raw}
    dev_freq = canon["developer"].value_counts(normalize=True)
    scaler = {c: (float(raw[c].fillna(medians[c]).mean()), float(raw[c].fillna(medians[c]).std(ddof=0)))
              for c in (F_ANDROID, F_AGE, F_RECENCY)}
    freq_col = canon["developer"].map(dev_freq).to_numpy(dtype=float)
    scaler[F_DEV] = (float(freq_col.mean()), float(freq_col.std(ddof=0)))
    return {"medians": medians, "scaler": scaler, "developer_freq": {k: float(v) for k, v in dev_freq.items()},
            "source": "computed from the raw Kaggle CSV"}


def processed_medians(scaled: pd.DataFrame, size_raw: np.ndarray, scaler: dict) -> dict:
    """A median survives median imputation and standardisation, so it can be read off the processed CSV."""
    medians = {F_SIZE: float(np.median(size_raw))}
    for col in (F_ANDROID, F_AGE, F_RECENCY):
        mean, scale = scaler[col]
        medians[col] = float(np.median(scaled[col].to_numpy()) * scale + mean)
    medians[F_ANDROID] = round(medians[F_ANDROID], 1)
    return medians


def _recover_grid_scaler(z, grid, mean, scale, anchors=None):
    """(mean, scale) of a standardised column whose raw values lie on a grid (0.1 or 1/N)."""
    zu = np.unique(z[np.isfinite(z)])
    if anchors:
        scale = (anchors[1] - anchors[0]) / (zu[1] - zu[0])
        mean = anchors[0] - zu[0] * scale
    for _ in range(3):
        a, b = np.polyfit(np.round((zu * scale + mean) / grid) * grid, zu, 1)
        mean, scale = -b / a, 1 / a
    err = np.abs((zu * scale + mean) / grid - np.round((zu * scale + mean) / grid)).max()
    return float(mean), float(scale), float(err)


def recover_from_processed(scaled: pd.DataFrame, size_raw: np.ndarray, n_apps: int, printed=None) -> dict:
    """Fallback without the raw CSV: no developer table, every developer is treated as unseen."""
    printed = printed or PHASE2_PRINTED_SCALER
    m_a, s_a, e_a = _recover_grid_scaler(scaled[F_ANDROID].to_numpy(), 0.1, *printed[F_ANDROID])
    m_d, s_d, e_d = _recover_grid_scaler(scaled[F_DEV].to_numpy(), 1 / n_apps, *printed[F_DEV],
                                         anchors=(1 / n_apps, 2 / n_apps))
    if max(e_a, e_d) > 0.01:
        raise ValueError("Could not recover the Phase 2 scalers; pass --raw with the Kaggle CSV.")
    scaler = {F_ANDROID: (m_a, s_a), F_DEV: (m_d, s_d), F_AGE: printed[F_AGE], F_RECENCY: printed[F_RECENCY]}
    return {"medians": processed_medians(scaled, size_raw, scaler), "scaler": scaler, "developer_freq": {},
            "source": "recovered from the processed CSV (no developer table)"}


# =============================================================== 4. metrics
def confusion(y_true, y_pred, k: int = 4) -> np.ndarray:
    return np.bincount(np.asarray(y_true) * k + np.asarray(y_pred), minlength=k * k).reshape(k, k)


def _macro_f1_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    denom = cm.sum(axis=-1) + cm.sum(axis=-2)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return (f1 * (denom > 0)).sum(axis=-1) / np.maximum((denom > 0).sum(axis=-1), 1)


def macro_f1(y_true, y_pred) -> float:
    return float(_macro_f1_cm(confusion(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion(y_true, y_pred)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    f1 = np.where(support + predicted > 0, 2 * tp / np.maximum(support + predicted, 1), 0)
    return {"macro_f1": macro_f1(y_true, y_pred), "accuracy": float(np.mean(y_true == y_pred)),
            "within_one_tier": float(np.mean(np.abs(y_true - y_pred) <= 1)),
            "per_tier_f1": {t: float(f1[i]) for i, t in enumerate(TIERS)},
            "per_tier_recall": {t: float(tp[i] / max(support[i], 1)) for i, t in enumerate(TIERS)},
            "per_tier_precision": {t: float(tp[i] / max(predicted[i], 1)) for i, t in enumerate(TIERS)}}


def paired_bootstrap_delta(y_true, pred_a, pred_b, n_resamples=BOOTSTRAP_RESAMPLES, seed=0, chunk=10) -> dict:
    """Δ = F1(a) - F1(b) with a 95% CI from a paired bootstrap."""
    y = np.asarray(y_true, np.int64)
    code_a, code_b = y * 4 + np.asarray(pred_a, np.int64), y * 4 + np.asarray(pred_b, np.int64)
    rng, deltas = np.random.default_rng(seed), []
    for start in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - start)
        idx = rng.integers(0, len(y), size=(m, len(y)))
        offs = (np.arange(m) * 16)[:, None]
        cm_a = np.bincount((code_a[idx] + offs).ravel(), minlength=m * 16).reshape(m, 4, 4)
        cm_b = np.bincount((code_b[idx] + offs).ravel(), minlength=m * 16).reshape(m, 4, 4)
        deltas.append(_macro_f1_cm(cm_a) - _macro_f1_cm(cm_b))
    deltas = np.concatenate(deltas)
    return {"delta": macro_f1(y, pred_a) - macro_f1(y, pred_b),
            "ci_low": float(np.percentile(deltas, 2.5)), "ci_high": float(np.percentile(deltas, 97.5))}


# =============================================================== 5. model registry
class Registry:
    """registry/models/rf-vN.joblib, registry/current.json (served version), registry/history.jsonl."""

    def __init__(self, root):
        self.root = Path(root)
        self.models, self.pointer, self.history = self.root / "models", self.root / "current.json", self.root / "history.jsonl"

    def _write(self, path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)  # atomic: the server never reads a half-written file

    def next_version(self) -> str:
        nums = [int(p.stem.split("v")[-1]) for p in self.models.glob("rf-v*.joblib")] if self.models.exists() else []
        return f"rf-v{max(nums, default=0) + 1}"

    def save(self, bundle: dict) -> Path:
        self.models.mkdir(parents=True, exist_ok=True)
        path = self.models / f"{bundle['version']}.joblib"
        fd, tmp = tempfile.mkstemp(dir=self.models, prefix=".tmp-", suffix=".joblib")
        os.close(fd)
        joblib.dump(bundle, tmp)  # uncompressed: a bigger file, but it loads in seconds, not minutes
        os.replace(tmp, path)
        return path

    def load(self, version: str) -> dict:
        # mmap: the tree arrays are read straight from the file instead of through a second copy in RAM
        return joblib.load(self.models / f"{version}.joblib", mmap_mode="r")

    def current(self) -> dict | None:
        return json.loads(self.pointer.read_text()) if self.pointer.exists() else None

    def current_version(self) -> str | None:
        cur = self.current()
        return cur["version"] if cur else None

    def publish(self, version: str):
        cur = self.current() or {}
        previous = cur.get("previous", []) + ([cur["version"]] if cur.get("version") else [])
        self._write(self.pointer, json.dumps({"version": version, "previous": previous,
                                              "published_at": datetime.now(timezone.utc).isoformat()}, indent=2))

    def rollback(self) -> str:
        cur = self.current()
        if not cur or not cur.get("previous"):
            raise RuntimeError("No previous version to roll back to.")
        self._write(self.pointer, json.dumps({"version": cur["previous"][-1], "previous": cur["previous"][:-1],
                                              "rolled_back_from": cur["version"]}, indent=2))
        return cur["previous"][-1]

    def log(self, entry: dict):
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.history, "a") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **entry}, default=str) + "\n")

    def read_history(self) -> list[dict]:
        return [json.loads(l) for l in self.history.read_text().splitlines() if l] if self.history.exists() else []
