"""Install-tier prediction service: API, feedback store and retraining (Milestone 4, Sections 5-6).

    uvicorn server:app --port 8000        run the API (docs at http://localhost:8000/docs)
    python server.py retrain [--force]    run one retraining job by hand
    python server.py rollback             serve the previous model version again

Environment: REGISTRY_DIR, DATA_DIR, DB_PATH, ADMIN_TOKEN, FEEDBACK_RATE_LIMIT ("20/3600"),
ENABLE_SCHEDULER ("1"), RETRAIN_N_ESTIMATORS (fewer trees for demos), CORS_ORIGINS ("*").
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from sklearn.ensemble import RandomForestClassifier

import common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("apscheduler").setLevel(logging.WARNING)  # hide routine scheduler messages
log = logging.getLogger("server")
ROOT = Path(__file__).resolve().parent
now = lambda: datetime.now(timezone.utc).isoformat()  # noqa: E731


# =============================================================== request / response models
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1, max_length=100)
    free: bool
    ad_supported: bool
    privacy_policy: bool
    content_rating: str
    size_mb: Optional[float] = Field(None, ge=0, le=100_000, description="null = Varies with device")
    minimum_android: str = Field("Varies", description="e.g. '5.0', or 'Varies'")
    developer: str = Field("", max_length=200)
    released: date
    last_updated: date
    observation_date: Optional[date] = Field(None, description="defaults to today")
    allow_feedback: bool = Field(False, description="store the inputs so installs can be reported later")

    @field_validator("last_updated")
    @classmethod
    def update_after_release(cls, v, info: ValidationInfo):
        if info.data.get("released") and v < info.data["released"]:
            raise ValueError("The last update cannot be before the release date.")
        return v


class WhatIf(BaseModel):
    change: str
    label: str
    high_or_above_delta: float  # change in confidence(High) + confidence(Very High)
    notable: bool               # False -> show "no notable effect"


class PredictResponse(BaseModel):
    tier: str
    installs: list[Optional[int]]
    confidence: dict[str, float]
    what_if: list[WhatIf]
    prediction_id: str
    model_version: str
    observation_date: date
    warnings: list[str]
    feedback_enabled: bool


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str = Field(min_length=1, max_length=64)
    actual_installs: int = Field(ge=0, le=100_000_000_000)
    reported_at: Optional[date] = Field(None, description="defaults to today")


# =============================================================== feedback store (SQLite)
class FeedbackStore:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, created_at TEXT, model_version TEXT,
                observation_date TEXT, predicted_tier INTEGER, raw_input TEXT);
            CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT UNIQUE REFERENCES predictions(id), actual_installs INTEGER,
                actual_tier INTEGER, reported_at TEXT, received_at TEXT);""")

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def run(self, sql, args=()):
        with self.lock, self._conn() as c:
            return c.execute(sql, args).lastrowid

    def one(self, sql, args=()):
        with self._conn() as c:
            row = c.execute(sql, args).fetchone()
        return dict(row) if row else None

    def all(self, sql, args=()):
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def training_samples(self):
        rows = self.all("SELECT f.actual_tier, f.reported_at, p.raw_input FROM feedback f JOIN predictions p"
                        " ON p.id = f.prediction_id WHERE p.raw_input IS NOT NULL ORDER BY f.reported_at, f.id")
        return [dict(r, raw_input=json.loads(r["raw_input"])) for r in rows]


# =============================================================== retraining job
def load_split(data_dir, split):
    """data/<split>_X.npy + <split>_y.npy, memory-mapped (older data/<split>.npz still accepted)."""
    d = Path(data_dir)
    if (d / f"{split}_X.npy").exists():
        return np.load(d / f"{split}_X.npy", mmap_mode="r"), np.load(d / f"{split}_y.npy")
    with np.load(d / f"{split}.npz") as z:
        return z["X"], z["y"]


def run_retrain(registry_dir, data_dir, db_path, force=False, n_estimators=None,
                bootstrap_resamples=C.BOOTSTRAP_RESAMPLES) -> dict:
    """Refit on training split + feedback; publish only if the candidate passes the gate."""
    reg = C.Registry(registry_dir)
    serving_version = reg.current_version()
    serving = reg.load(serving_version)
    samples = FeedbackStore(db_path).training_samples()
    new = len(samples) - int(serving.get("n_feedback") or 0)
    result = {"event": "retrain", "serving_version": serving_version, "n_feedback": len(samples), "n_new": new}
    if not samples or (new <= 0 and not force):
        reg.log(dict(result, published=False, reason="no new feedback"))
        return dict(result, published=False, reason="no new feedback")

    # feedback features: the report date is the observation date (like the scrape date in training)
    canon = pd.concat([C.canonical_from_request(dict(s["raw_input"], observation_date=s["reported_at"]))
                       for s in samples], ignore_index=True)
    X_fb = serving["pipeline"].transform(canon)
    y_fb = np.array([s["actual_tier"] for s in samples], np.int8)
    k = max(1, round(len(samples) * C.RECENT_HOLDOUT_FRACTION)) if len(samples) >= C.MIN_FEEDBACK_FOR_WEIGHT_SEARCH else 0
    X_te, y_te = load_split(data_dir, "test")
    params = dict(serving["rf_params"], **({"n_estimators": n_estimators} if n_estimators else {}))

    # Memory: score the serving model now, then drop it, so it is never in RAM next to a new forest.
    pred_s = serving["model"].predict(X_te)
    pred_s_recent = serving["model"].predict(X_fb[-k:]) if k else None
    del serving["model"]
    gc.collect()

    # Memory: one combined copy of the training data (training split + feedback used for fitting).
    X_extra, y_extra = (X_fb[:-k], y_fb[:-k]) if k else (X_fb, y_fb)
    X_tr, y_tr = load_split(data_dir, "train")  # memory-mapped: read from disk, not copied into RAM
    n_train = len(y_tr)
    X_all = np.vstack([X_tr, X_extra])
    y_all = np.concatenate([y_tr, y_extra])
    del X_tr, y_tr
    gc.collect()

    def fit(w):
        weights = np.concatenate([np.ones(n_train), np.full(len(y_extra), float(w))])
        return RandomForestClassifier(**params).fit(X_all, y_all, sample_weight=weights)

    recent = None
    if k:  # newest 20% of feedback held out to choose w; only the best forest is kept
        weight, cand_f1, candidate = None, -1.0, None
        for w in C.FEEDBACK_WEIGHTS:
            m = fit(w)
            score = C.macro_f1(y_fb[-k:], m.predict(X_fb[-k:]))
            if score > cand_f1:
                weight, cand_f1, candidate = w, score, m
            del m
            gc.collect()
        recent = {"n": k, "candidate_f1": cand_f1, "delta": cand_f1 - C.macro_f1(y_fb[-k:], pred_s_recent)}
    else:
        weight = C.DEFAULT_FEEDBACK_WEIGHT
        candidate = fit(weight)
    n_used = len(y_extra)
    del X_all, y_all
    gc.collect()

    pred_c = candidate.predict(X_te)
    gate = C.paired_bootstrap_delta(y_te, pred_c, pred_s, n_resamples=bootstrap_resamples)
    publish = gate["ci_high"] >= 0 and (recent is None or recent["delta"] >= 0)
    result.update(feedback_weight=weight, gate_test=gate, published=publish,
                  recent=recent or f"skipped (fewer than {C.MIN_FEEDBACK_FOR_WEIGHT_SEARCH} feedback samples)")
    if publish:
        version = reg.next_version()
        reg.save(dict(serving, version=version, model=candidate, rf_params=params, n_feedback=len(samples),
                      train_size=int(n_train + n_used), feedback_weight=weight,
                      metrics=C.evaluate(y_te, pred_c), trained_at=now(), parent_version=serving_version))
        reg.publish(version)
        result["new_version"] = version
    else:
        result["reason"] = "candidate not better on the test set or the recent feedback"
    reg.log(result)
    return result


# =============================================================== the app
@dataclass
class Settings:
    registry_dir: str = os.getenv("REGISTRY_DIR", "registry")
    data_dir: str = os.getenv("DATA_DIR", "data")
    db_path: str = os.getenv("DB_PATH", "data/feedback.db")
    admin_token: str = os.getenv("ADMIN_TOKEN", "")
    feedback_rate: str = os.getenv("FEEDBACK_RATE_LIMIT", "20/3600")
    enable_scheduler: bool = os.getenv("ENABLE_SCHEDULER", "1") == "1"
    retrain_n_estimators: int = int(os.getenv("RETRAIN_N_ESTIMATORS", "0"))
    cors_origins: list = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    sync_load: bool = False


def error(status, fld, message):
    return HTTPException(status, detail=[{"field": fld, "message": message}])


def create_app(s: Settings | None = None) -> FastAPI:
    s = s or Settings()
    registry, store = C.Registry(s.registry_dir), FeedbackStore(s.db_path)
    state = {"bundle": None, "error": None}
    retrain_status = {"running": False, "last_trigger": None, "last_finished": None,
                      "last_result": None, "last_error": None}
    retrain_lock, load_lock = threading.Lock(), threading.Lock()
    limit_n, limit_s = (float(x) for x in s.feedback_rate.split("/"))
    calls = defaultdict(deque)

    def load_model():  # also used to hot-swap a newly published version
        if not load_lock.acquire(blocking=False):
            return  # a load is already running; the 30 s check simply waits for the next round
        try:
            version = registry.current_version()
            if version is None:
                state["error"] = "No published model in the registry."
            elif state["bundle"] is None or state["bundle"]["version"] != version:
                log.info("Loading %s ...", version)
                t0 = time.time()
                bundle = registry.load(version)
                bundle["model"].n_jobs = 1  # 3 rows per request: threads cost more than they save
                state["bundle"], state["error"] = bundle, None
                log.info("Serving %s (loaded in %.0f s)", version, time.time() - t0)
        finally:
            load_lock.release()

    def start_retrain(trigger, force=False):
        if not retrain_lock.acquire(blocking=False):
            return False
        retrain_status.update(running=True, last_trigger=trigger)

        def job():
            cmd = [sys.executable, str(ROOT / "server.py"), "retrain", "--registry", s.registry_dir,
                   "--data-dir", s.data_dir, "--db", s.db_path] + (["--force"] if force else [])
            if s.retrain_n_estimators:
                cmd += ["--n-estimators", str(s.retrain_n_estimators)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True)
                retrain_status["last_result"] = json.loads(p.stdout) if p.returncode == 0 else None
                retrain_status["last_error"] = None if p.returncode == 0 else p.stderr[-2000:]
                load_model()
            finally:
                retrain_status.update(running=False, last_finished=now())
                retrain_lock.release()

        threading.Thread(target=job, daemon=True).start()
        return True

    def check_volume():
        b = state["bundle"]
        count = store.one("SELECT COUNT(*) AS n FROM feedback")["n"]
        if b and count - int(b.get("n_feedback") or 0) >= C.RETRAIN_SAMPLE_TRIGGER:
            start_retrain("1,000 new feedback samples")

    @asynccontextmanager
    async def lifespan(app):
        if s.sync_load:
            load_model()
        else:
            threading.Thread(target=load_model, daemon=True).start()
        sched = None
        if s.enable_scheduler:
            from apscheduler.schedulers.background import BackgroundScheduler
            sched = BackgroundScheduler(timezone="UTC")
            sched.add_job(lambda: start_retrain("weekly schedule"), "cron", day_of_week="sun", hour=3)
            sched.add_job(check_volume, "interval", hours=1)
            sched.add_job(load_model, "interval", seconds=30)  # picks up rollbacks / new versions
            sched.start()
        yield
        if sched:
            sched.shutdown(wait=False)

    app = FastAPI(title="Google Play Install Tier Service", lifespan=lifespan)
    app.state.reload = load_model
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def on_invalid(request, exc):
        return JSONResponse(status_code=422, content={"detail": [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
             "message": e["msg"].removeprefix("Value error, ")} for e in exc.errors()]})

    def model():
        if state["bundle"] is None:
            raise HTTPException(503, detail="The model is still loading. Try again in a moment.")
        return state["bundle"]

    def admin(token):
        if not s.admin_token or token != s.admin_token:
            raise HTTPException(401, detail="Admin token missing or wrong (set ADMIN_TOKEN).")

    @app.get("/health")
    def health():
        if state["bundle"] is None:
            return JSONResponse(status_code=503, content={"status": "loading", "error": state["error"]})
        return {"status": "ready", "model_version": state["bundle"]["version"]}

    @app.get("/schema")
    def schema():
        b = model()
        p = b["pipeline"]
        f = lambda name, label, group, typ, **kw: {"name": name, "label": label, "group": group, "type": typ, **kw}  # noqa: E731
        return {"model_version": b["version"], "observation_date_default": date.today().isoformat(),
                "fields": [
                    f("category", "Category", "App", "enum", required=True, allowed=p.categories),
                    f("content_rating", "Content rating", "App", "enum", required=True, allowed=p.content_ratings),
                    f("developer", "Developer name", "App", "string", required=False),
                    f("free", "Free", "Monetisation", "boolean", required=True),
                    f("ad_supported", "Contains ads", "Monetisation", "boolean", required=True),
                    f("size_mb", "App size (MB)", "Technical", "number", required=False, min=0,
                      null_means="Varies with device"),
                    f("minimum_android", "Minimum Android", "Technical", "enum", required=False,
                      allowed=[f"{v:.1f}" for v in p.android_versions] + ["Varies"], default="Varies"),
                    f("released", "Release date", "Timeline", "date", required=True),
                    f("last_updated", "Last update", "Timeline", "date", required=True),
                    f("privacy_policy", "Has a privacy policy", "Policy", "boolean", required=True),
                    f("allow_feedback", "Let me report actual installs later", "Policy", "boolean", default=False)],
                "tiers": [{"name": t, "min": C.TIER_RANGES[t][0], "max": C.TIER_RANGES[t][1]} for t in C.TIERS]}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        b = model()
        pipe = b["pipeline"]
        category = C.merge_category(req.category)
        if category not in pipe.categories:
            raise error(422, "category", f"Unknown category '{req.category}'. See GET /schema.")
        if req.content_rating not in pipe.content_ratings:
            raise error(422, "content_rating", f"Unknown content rating '{req.content_rating}'.")
        if req.minimum_android != "Varies" and np.isnan(C.parse_android_version(req.minimum_android)):
            raise error(422, "minimum_android", "Use a version such as '5.0', or 'Varies'.")

        obs = req.observation_date or date.today()
        raw = dict(req.model_dump(mode="json", exclude={"observation_date", "allow_feedback"}), category=category)
        base = dict(raw, observation_date=obs.isoformat())
        variants = [("ad_supported", dict(base, ad_supported=not req.ad_supported),
                     "Remove ads" if req.ad_supported else "Add ads"),
                    ("free", dict(base, free=not req.free), "Switch to paid" if req.free else "Switch to free")]
        canon = pd.concat([C.canonical_from_request(r) for r in [base] + [v[1] for v in variants]], ignore_index=True)
        proba = np.zeros((3, 4))
        proba[:, b["model"].classes_.astype(int)] = b["model"].predict_proba(pipe.transform(canon))  # one call
        tier_idx, high0 = int(np.argmax(proba[0])), proba[0, 2] + proba[0, 3]
        what_if = [WhatIf(change=f"{n}={str(v[n]).lower()}", label=label,
                          high_or_above_delta=round(float(p[2] + p[3] - high0), 4),
                          notable=abs(p[2] + p[3] - high0) >= C.WHAT_IF_NOTABLE)
                   for (n, v, label), p in zip(variants, proba[1:])]
        pid = "p-" + uuid.uuid4().hex[:12]
        store.run("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                  (pid, now(), b["version"], obs.isoformat(), tier_idx,
                   json.dumps(raw) if req.allow_feedback else None))
        tier = C.TIERS[tier_idx]
        return PredictResponse(tier=tier, installs=C.TIER_RANGES[tier],
                               confidence={t: round(float(proba[0, i]), 4) for i, t in enumerate(C.TIERS)},
                               what_if=what_if, prediction_id=pid, model_version=b["version"],
                               observation_date=obs, warnings=pipe.warnings_for(canon.iloc[[0]]),
                               feedback_enabled=req.allow_feedback)

    @app.post("/feedback")
    def feedback(req: FeedbackRequest, request: Request):
        q, t = calls[request.client.host if request.client else "?"], time.monotonic()
        while q and t - q[0] > limit_s:
            q.popleft()
        if len(q) >= limit_n:
            raise HTTPException(429, detail="Too many reports from this address. Try again later.")
        q.append(t)
        pred = store.one("SELECT * FROM predictions WHERE id=?", (req.prediction_id,))
        if pred is None:
            raise error(404, "prediction_id", "No prediction with this id.")
        if pred["raw_input"] is None:
            raise error(409, "prediction_id", "This prediction was made without allow_feedback, so it cannot be used.")
        if store.one("SELECT 1 FROM feedback WHERE prediction_id=?", (req.prediction_id,)):
            raise error(409, "prediction_id", "Installs were already reported for this prediction.")
        reported = req.reported_at or date.today()
        if reported > date.today():
            raise error(422, "reported_at", "The report date cannot be in the future.")
        if reported < date.fromisoformat(json.loads(pred["raw_input"])["released"]):
            raise error(422, "reported_at", "Installs can only be reported after the release date.")
        actual = int(C.installs_to_tier([req.actual_installs])[0])
        fid = store.run("INSERT INTO feedback (prediction_id, actual_installs, actual_tier, reported_at, received_at)"
                        " VALUES (?,?,?,?,?)", (req.prediction_id, req.actual_installs, actual, reported.isoformat(), now()))
        check_volume()
        return {"status": "stored", "feedback_id": fid, "actual_tier": C.TIERS[actual],
                "predicted_tier": C.TIERS[pred["predicted_tier"]],
                "prediction_was_correct": actual == pred["predicted_tier"]}

    @app.get("/model")
    def model_info():
        b = model()
        pairs = store.all("SELECT f.actual_tier AS a, p.predicted_tier AS p FROM feedback f"
                          " JOIN predictions p ON p.id = f.prediction_id")
        live = {"n_feedback": len(pairs)}
        if pairs:
            live["macro_f1"] = round(C.macro_f1([r["a"] for r in pairs], [r["p"] for r in pairs]), 4)
        return {"model_version": b["version"], "trained_at": b["trained_at"], "train_size": b["train_size"],
                "training_data": b.get("data_file"), "n_feedback_in_training": b.get("n_feedback", 0),
                "feedback_weight": b.get("feedback_weight"), "test_metrics": b["metrics"],
                "developer_table_loaded": b["pipeline"].has_developer_table,
                "live_monitoring": live, "previous_versions": (registry.current() or {}).get("previous", []),
                "retraining_running": retrain_status["running"]}

    @app.post("/admin/retrain", status_code=202)
    def admin_retrain(force: bool = False, x_admin_token: str | None = Header(None)):
        admin(x_admin_token)
        if not start_retrain("manual", force):
            raise HTTPException(409, detail="A retraining run is already in progress.")
        return {"status": "started"}

    @app.get("/admin/retrain/status")
    def admin_status(x_admin_token: str | None = Header(None)):
        admin(x_admin_token)
        return retrain_status

    @app.get("/admin/history")
    def admin_history(x_admin_token: str | None = Header(None)):
        admin(x_admin_token)
        return registry.read_history()

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Retrain or roll back the served model.")
    ap.add_argument("command", choices=["retrain", "rollback"])
    ap.add_argument("--registry", default=os.getenv("REGISTRY_DIR", "registry"))
    ap.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    ap.add_argument("--db", default=os.getenv("DB_PATH", "data/feedback.db"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-estimators", type=int)
    a = ap.parse_args()
    if a.command == "rollback":
        print("Now serving", C.Registry(a.registry).rollback())
    else:
        print(json.dumps(run_retrain(a.registry, a.data_dir, a.db, a.force, a.n_estimators), default=str))
else:
    app = create_app()
