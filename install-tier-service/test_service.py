"""Essential tests (pytest -q). Runs on SYNTHETIC data in about 10 s.

Synthetic raw Kaggle-format CSV -> a copy of the Phase 2 + Untitled8 notebook code -> processed CSV
-> train.py -> server. The key check: the server rebuilds the processed CSV's features row by row.
"""
import os
import re
import shutil
import tempfile
import time

os.environ.update(ENABLE_SCHEDULER="0", DB_PATH=os.path.join(tempfile.mkdtemp(), "module.db"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import common as C  # noqa: E402
import server  # noqa: E402
import train  # noqa: E402

TREES = 15


# ------------------------------------------------------------------ synthetic data
def make_raw(rows=8000, seed=1) -> pd.DataFrame:
    """Random apps in the Kaggle column layout (48 categories -> 46 after merging)."""
    cats = ["Adventure", "Tools", "Productivity", "Communication", "Social", "Libraries & Demo", "Lifestyle",
            "Personalization", "Racing", "Maps & Navigation", "Travel & Local", "Food & Drink",
            "Books & Reference", "Medical", "Puzzle", "Entertainment", "Arcade", "Auto & Vehicles",
            "Photography", "Health & Fitness", "Education", "Shopping", "Board", "Music & Audio", "Sports",
            "Beauty", "Business", "Educational", "Finance", "News & Magazines", "Casual", "Art & Design",
            "House & Home", "Card", "Events", "Trivia", "Weather", "Strategy", "Word", "Video Players & Editors",
            "Action", "Simulation", "Music", "Dating", "Role Playing", "Casino", "Comics", "Parenting"]
    rng = np.random.default_rng(seed)
    scraped = pd.Timestamp("2021-06-15")
    free, ads, privacy = rng.random(rows) < 0.97, rng.random(rows) < 0.5, rng.random(rows) < 0.8
    dev_pop = rng.zipf(1.6, rows // 3).clip(max=500)
    dev = rng.integers(0, rows // 3, rows)
    age = rng.integers(30, 3650, rows)
    upd = (rng.random(rows) * age).astype(int)
    score = 1.2 * ads + privacy + 0.8 * free + 0.4 * np.log(dev_pop[dev]) + 0.0004 * age - 0.002 * upd
    installs = np.exp(6.5 + 1.4 * (score + rng.normal(0, 1.6, rows))).astype(np.int64)
    size = np.array([f"{x:.1f}M" if x >= 1 else f"{x * 1024:.0f}k" for x in np.exp(rng.normal(2.8, 1.2, rows))], object)
    size[rng.random(rows) < 0.03] = "Varies with device"
    rel = (scraped - pd.to_timedelta(age, unit="D")).strftime("%b %d, %Y").to_numpy().astype(object)
    rel[rng.random(rows) < 0.03] = np.nan
    rel[rng.random(rows) < 0.005] = "Jul 01, 2021"  # released after the scrape
    developer = np.array([f"Developer {d}" for d in dev], object)
    developer[rng.random(rows) < 0.002] = None
    return pd.DataFrame({
        "Category": rng.choice(cats, rows), "Rating Count": rng.integers(0, 9999, rows), "Installs": "",
        "Minimum Installs": installs // 2, "Maximum Installs": installs, "Free": free, "Size": size,
        "Minimum Android": rng.choice(["4.0.3 and up", "4.1 and up", "5.0 and up", "8.0 and up",
                                       "4.1 - 7.1.1", "Varies with device"], rows),
        "Developer Id": developer, "Released": rel,
        "Last Updated": (scraped - pd.to_timedelta(upd, unit="D")).strftime("%b %d, %Y"),
        "Content Rating": rng.choice(["Everyone", "Teen", "Mature 17+", "Everyone 10+"], rows, p=[.85, .07, .04, .04]),
        "Privacy Policy": np.where(privacy, "https://example.com/privacy", None), "Ad Supported": ads,
        "Scraped Time": rng.choice(["2021-06-15 20:19:35", "2021-06-16 01:02:03"], rows)})


def notebooks_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Independent copy of google_playstore_phase2_preprocessing.ipynb + Untitled8.ipynb (no common.py)."""
    work = df.dropna(subset=["Maximum Installs"]).copy()
    b = {True: 1, False: 0, "True": 1, "False": 0}
    work["Free"] = work["Free"].map(b).fillna(0).astype("int8")
    work["Ad Supported"] = work["Ad Supported"].map(b).fillna(0).astype("int8")

    def size_to_mb(v):
        s = str(v).strip().replace(",", "").lower()
        m = None if pd.isna(v) or "varies" in s else re.search(r"([0-9]*\.?[0-9]+)\s*([kmg]?)", s)
        return np.nan if not m else float(m.group(1)) / 1024 if m.group(2) == "k" else \
            float(m.group(1)) * 1024 if m.group(2) == "g" else float(m.group(1))

    work["Size_MB"] = work["Size"].apply(size_to_mb)
    work["Size_MB"] = work["Size_MB"].fillna(work["Size_MB"].median())
    work["Size_Tier"] = pd.cut(work["Size_MB"], bins=[-np.inf, 10, 30, 100, 300, np.inf],
                               labels=["Tiny", "Small", "Medium", "Large", "Very_Large"], right=False)
    android = work["Minimum Android"].apply(
        lambda v: np.nan if pd.isna(v) or "varies" in str(v).lower()
        else float(re.search(r"(\d+(?:\.\d+)?)", str(v)).group(1)))
    work["Minimum_Android_Numeric"] = android.fillna(android.median())
    dev = work["Developer Id"].fillna("__MISSING_DEVELOPER__")
    work["Developer_Frequency"] = dev.map(dev.value_counts(normalize=True))
    for c in ["Released", "Last Updated", "Scraped Time"]:
        work[c] = pd.to_datetime(work[c], errors="coerce")
    month = 60 * 60 * 24 * 30.44
    work["App_Age_Months"] = (work["Scraped Time"] - work["Released"]).dt.total_seconds() / month
    work.loc[work["App_Age_Months"] < 0, "App_Age_Months"] = np.nan
    work["Months_Since_Last_Update"] = (work["Scraped Time"] - work["Last Updated"]).dt.total_seconds().abs() / month
    for c in ["App_Age_Months", "Months_Since_Last_Update"]:
        work[c] = work[c].fillna(work[c].median())
    work["Is_Everyone"] = work["Content Rating"].fillna("").str.strip().str.lower().eq("everyone").astype("int8")
    work["Has_Privacy_Policy"] = work["Privacy Policy"].notna().astype("int8")
    work["Category"] = work["Category"].fillna("Missing").replace({"Educational": "Education", "Music": "Music & Audio"})
    work["Installs_Class"] = pd.cut(work["Maximum Installs"], [-np.inf, 1e4, 1e5, 1e6, np.inf],
                                    labels=["Low", "Moderate", "High", "Very High"])
    work = work[["Free", "Ad Supported", "Size_MB", "Minimum_Android_Numeric", "Developer_Frequency",
                 "App_Age_Months", "Months_Since_Last_Update", "Is_Everyone", "Has_Privacy_Policy",
                 "Category", "Size_Tier", "Installs_Class", "Rating Count"]]
    work = pd.get_dummies(work, columns=["Category", "Size_Tier"], prefix=["Category", "Size"], dtype="int8")
    cont = ["Minimum_Android_Numeric", "Developer_Frequency", "App_Age_Months", "Months_Since_Last_Update"]
    work[cont] = StandardScaler().fit_transform(work[cont])
    return work


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained")
    raw = make_raw()
    raw.to_csv(root / "raw.csv", index=False)
    notebooks_copy(raw).to_csv(root / "processed.csv", index=False)
    train.main(["--data", str(root / "processed.csv"), "--raw", str(root / "raw.csv"), "--n-estimators", str(TREES),
                "--registry", str(root / "registry"), "--data-dir", str(root / "data")])
    return root


@pytest.fixture
def client(trained, tmp_path):
    shutil.copytree(trained / "registry", tmp_path / "registry")  # each test gets its own registry copy
    s = server.Settings(registry_dir=str(tmp_path / "registry"), data_dir=str(trained / "data"),
                        db_path=str(tmp_path / "f.db"), admin_token="t", feedback_rate="1000/60",
                        enable_scheduler=False, sync_load=True, retrain_n_estimators=TREES)
    with TestClient(server.create_app(s)) as c:
        yield c


APP = {"category": "Education", "free": True, "ad_supported": True, "privacy_policy": True,
       "content_rating": "Everyone", "size_mb": 25, "minimum_android": "5.0", "developer": "Developer 1",
       "released": "2020-01-15", "last_updated": "2021-03-01", "observation_date": "2021-06-15",
       "allow_feedback": True}


def pipeline(trained):
    reg = C.Registry(trained / "registry")
    return reg.load(reg.current_version())["pipeline"]


# ------------------------------------------------------------------ 1. features = Phase 2
def test_server_features_match_processed_csv(trained):
    pipe = pipeline(trained)
    ours = pipe.transform(C.canonical_from_dataset(pd.read_csv(trained / "raw.csv", low_memory=False)))
    theirs = pd.read_csv(trained / "processed.csv")[pipe.feature_names].to_numpy(float)
    j = pipe.index_of(C.F_SIZE)
    theirs[:, j] = (np.float32(theirs[:, j]) - np.float32(pipe.size_mean)) / np.float32(pipe.size_std)
    assert len(pipe.feature_names) == 60 and len(pipe.categories) == 46
    np.testing.assert_allclose(ours, theirs, atol=1e-4)


@pytest.mark.parametrize("mb,col", [(9.99, "Size_Tiny"), (10, "Size_Small"), (30, "Size_Medium"),
                                    (100, "Size_Large"), (300, "Size_Very_Large")])
def test_size_tier_edges(mb, col):
    assert C.SIZE_TIER_COLUMNS[C.size_tier_index(np.array([mb]))[0]] == col


@pytest.mark.parametrize("installs,tier", [(10_000, "Low"), (10_001, "Moderate"), (1_000_000, "High"),
                                           (1_000_001, "Very High")])
def test_install_tier_edges(installs, tier):
    assert C.TIERS[C.installs_to_tier([installs])[0]] == tier


# ------------------------------------------------------------------ 2. API
def test_predict_output(client):
    assert client.get("/health").json() == {"status": "ready", "model_version": "rf-v1"}
    assert len(client.get("/schema").json()["fields"][0]["allowed"]) == 46
    d = client.post("/predict", json=APP).json()
    assert sum(d["confidence"].values()) == pytest.approx(1, abs=1e-3)
    assert d["tier"] == max(d["confidence"], key=d["confidence"].get)
    assert d["installs"] == C.TIER_RANGES[d["tier"]] and len(d["what_if"]) == 2


@pytest.mark.parametrize("change", [{"size_mb": None, "minimum_android": "Varies"}, {"category": "Educational"},
                                    {"size_mb": 10}, {"developer": ""}])
def test_edge_inputs_accepted(client, change):
    assert client.post("/predict", json=dict(APP, **change)).status_code == 200


@pytest.mark.parametrize("change,fld", [({"size_mb": -1}, "size_mb"), ({"last_updated": "2019-01-01"}, "last_updated"),
                                        ({"category": "Nope"}, "category"), ({"content_rating": "PG"}, "content_rating")])
def test_invalid_input_is_422(client, change, fld):
    r = client.post("/predict", json=dict(APP, **change))
    assert r.status_code == 422 and fld in [e["field"] for e in r.json()["detail"]]


def test_warnings(client):
    assert any("not seen" in w for w in client.post("/predict", json=dict(APP, developer="New Studio")).json()["warnings"])


def test_503_before_model_loaded(tmp_path):
    s = server.Settings(registry_dir=str(tmp_path / "none"), db_path=str(tmp_path / "f.db"),
                        enable_scheduler=False, sync_load=True)
    with TestClient(server.create_app(s)) as c:
        assert c.get("/health").status_code == 503 and c.post("/predict", json=APP).status_code == 503


# ------------------------------------------------------------------ 3. feedback + retraining
def test_feedback_rules(client):
    pid = client.post("/predict", json=APP).json()["prediction_id"]
    post = lambda **b: client.post("/feedback", json={"prediction_id": pid, "actual_installs": 5, **b}).status_code  # noqa: E731
    assert post(reported_at="2019-01-01") == 422           # before release
    assert post(actual_installs=-5) == 422                  # negative
    assert client.post("/feedback", json={"prediction_id": pid, "actual_installs": 250_000}).json()["actual_tier"] == "High"
    assert post() == 409                                    # second report
    assert client.post("/feedback", json={"prediction_id": "p-x", "actual_installs": 1}).status_code == 404
    no_consent = client.post("/predict", json=dict(APP, allow_feedback=False)).json()["prediction_id"]
    assert client.post("/feedback", json={"prediction_id": no_consent, "actual_installs": 1}).status_code == 409


def test_retrain_publishes_or_rejects_and_hot_swaps(client, tmp_path):
    for i in range(5):
        pid = client.post("/predict", json=dict(APP, size_mb=5 + i)).json()["prediction_id"]
        client.post("/feedback", json={"prediction_id": pid, "actual_installs": 500_000, "reported_at": "2021-06-15"})
    assert client.post("/admin/retrain").status_code == 401
    assert client.post("/admin/retrain", headers={"X-Admin-Token": "t"}).status_code == 202
    for _ in range(120):
        st = client.get("/admin/retrain/status", headers={"X-Admin-Token": "t"}).json()
        if st["last_finished"]:
            break
        time.sleep(0.5)
    assert st["last_error"] is None, st["last_error"]
    res = st["last_result"]
    expected = res["new_version"] if res["published"] else "rf-v1"
    assert client.get("/health").json()["model_version"] == expected
    assert C.Registry(tmp_path / "registry").read_history()[-1]["event"] == "retrain"


def test_gate_rejects_worse_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 5000)
    good = np.where(rng.random(5000) < 0.8, y, rng.integers(0, 4, 5000))
    assert C.paired_bootstrap_delta(y, rng.integers(0, 4, 5000), good, n_resamples=200)["ci_high"] < 0
