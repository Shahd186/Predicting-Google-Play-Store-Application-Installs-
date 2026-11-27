"""Check a RUNNING server end to end and print PASS / FAIL for each check.

    python check_service.py                      # predictions, validation, schema, model info
    python check_service.py --feedback           # + feedback rules (writes test reports to the server's DB)
    python check_service.py --feedback --retrain --token secret   # + a full retraining run (slow)

Start the server for --feedback with a separate database, so test reports never become training data:
    set DB_PATH=data\\check_feedback.db  (cmd)   then   python -m uvicorn server:app --port 8000
Uses only the Python standard library.
"""
import argparse
import json
import time
import urllib.error
import urllib.request

APP = {"category": "Education", "free": True, "ad_supported": True, "privacy_policy": True,
       "content_rating": "Everyone", "size_mb": 25, "minimum_android": "5.0", "developer": "",
       "released": "2024-01-15", "last_updated": "2026-09-01", "allow_feedback": True}
results = []


def call(method, path, body=None, token=None, timeout=120):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"X-Admin-Token": token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")
    except (TimeoutError, ConnectionError, urllib.error.URLError) as e:
        return 0, {"detail": f"no answer from the server ({e})"}  # busy or not running


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def fields(body):
    return [e.get("field") for e in body.get("detail", [])] if isinstance(body, dict) and isinstance(body.get("detail"), list) else []


def main():
    global BASE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--token", default="")
    a = ap.parse_args()
    BASE = a.url.rstrip("/")

    print("== 1. Service and model")
    s, h = call("GET", "/health")
    check("GET /health is ready", s == 200 and h.get("status") == "ready", f"{s} {h}")
    if s != 200:
        print("The model is not loaded yet (or the server is not running). Wait for 'Serving rf-v1' and retry.")
        return
    s, sc = call("GET", "/schema")
    cats = next((f["allowed"] for f in sc.get("fields", []) if f["name"] == "category"), [])
    check("GET /schema lists 46 categories and 4 tiers", s == 200 and len(cats) == 46 and len(sc.get("tiers", [])) == 4)
    s, m = call("GET", "/model")
    check("GET /model reports the model", s == 200 and "test_metrics" in m,
          f"version {m.get('model_version')}, test macro-F1 {m.get('test_metrics', {}).get('macro_f1', 0):.4f}")

    print("\n== 2. Predictions")
    s, p = call("POST", "/predict", APP)
    conf = p.get("confidence", {}) if s == 200 else {}
    check("valid app -> 200", s == 200, f"tier {p.get('tier')}, installs {p.get('installs')}" if s == 200 else str(p))
    check("4 confidences summing to 1", len(conf) == 4 and abs(sum(conf.values()) - 1) < 1e-3, str(conf))
    check("tier = highest confidence", bool(conf) and p["tier"] == max(conf, key=conf.get))
    check("2 what-if entries", len(p.get("what_if", [])) == 2,
          "; ".join(f"{w['label']}: {w['high_or_above_delta']:+.3f}" for w in p.get("what_if", [])))
    s2, p2 = call("POST", "/predict", APP)
    check("same input -> same prediction", s2 == 200 and p2.get("confidence") == conf)
    for name, change in [("size 'Varies' (null)", {"size_mb": None}), ("Android 'Varies'", {"minimum_android": "Varies"}),
                         ("merged category 'Educational'", {"category": "Educational"}),
                         ("size exactly 10 MB", {"size_mb": 10}), ("unknown developer", {"developer": "Brand New Studio"})]:
        s, r = call("POST", "/predict", dict(APP, **change))
        check(f"edge input: {name} -> 200", s == 200, "; ".join(r.get("warnings", [])) if s == 200 else str(r))

    print("\n== 3. Input validation (each must be rejected with 422 naming the field)")
    for name, change, fld in [("negative size", {"size_mb": -3}, "size_mb"),
                              ("update before release", {"last_updated": "2023-01-01"}, "last_updated"),
                              ("unknown category", {"category": "string"}, "category"),
                              ("unknown content rating", {"content_rating": "string"}, "content_rating"),
                              ("bad Android version", {"minimum_android": "banana"}, "minimum_android"),
                              ("unexpected field", {"rating": 4.5}, "rating")]:
        s, r = call("POST", "/predict", dict(APP, **change))
        check(f"{name} -> 422", s == 422 and fld in fields(r), f"{s} {fields(r)}")
    body = dict(APP)
    del body["category"]
    s, r = call("POST", "/predict", body)
    check("missing category -> 422", s == 422 and "category" in fields(r))

    print("\n== 4. A few profiles side by side (no right answer: check they look sensible)")
    profiles = {
        "new small paid app, no ads": dict(APP, free=False, ad_supported=False, size_mb=5,
                                           released="2026-09-01", last_updated="2026-09-01"),
        "your example": APP,
        "older free game with ads, updated often": dict(APP, category="Casual", size_mb=80,
                                                        released="2018-03-01", last_updated="2026-11-01"),
    }
    for name, body in profiles.items():
        s, r = call("POST", "/predict", dict(body, allow_feedback=False))
        print(f"      {name:42s} -> {r.get('tier', s)}  {r.get('confidence', '')}")

    if a.feedback:
        print("\n== 5. Feedback rules (writes test reports to the server's database)")
        pid = call("POST", "/predict", APP)[1]["prediction_id"]
        s, r = call("POST", "/feedback", {"prediction_id": pid, "actual_installs": 250000})
        check("valid report -> stored, tier High", s == 200 and r.get("actual_tier") == "High", str(r))
        check("second report -> 409", call("POST", "/feedback", {"prediction_id": pid, "actual_installs": 5})[0] == 409)
        check("unknown prediction -> 404", call("POST", "/feedback", {"prediction_id": "p-unknown", "actual_installs": 5})[0] == 404)
        pid2 = call("POST", "/predict", APP)[1]["prediction_id"]
        check("negative installs -> 422", call("POST", "/feedback", {"prediction_id": pid2, "actual_installs": -1})[0] == 422)
        check("report before release -> 422",
              call("POST", "/feedback", {"prediction_id": pid2, "actual_installs": 5, "reported_at": "2020-01-01"})[0] == 422)
        pid3 = call("POST", "/predict", dict(APP, allow_feedback=False))[1]["prediction_id"]
        check("prediction without consent -> 409", call("POST", "/feedback", {"prediction_id": pid3, "actual_installs": 5})[0] == 409)
        check("/model counts the feedback", call("GET", "/model")[1].get("live_monitoring", {}).get("n_feedback", 0) >= 1)

    if a.retrain:
        print("\n== 6. Retraining (refits on the full training set: can take 15-30 minutes)")
        check("retrain without token -> 401", call("POST", "/admin/retrain")[0] == 401)
        s, r = call("POST", "/admin/retrain?force=true", token=a.token)
        check("retrain started", s == 202, f"{s} {r}")
        t0 = time.time()
        while s == 202:
            time.sleep(30)
            code, st = call("GET", "/admin/retrain/status", token=a.token, timeout=60)
            if code != 200:  # the PC is busy (often out of memory): keep waiting instead of failing
                print(f"      ... {time.time() - t0:.0f}s, server busy, no answer yet")
                continue
            print(f"      ... {time.time() - t0:.0f}s, running={st.get('running')}")
            if not st.get("running") and st.get("last_finished"):
                res = st.get("last_result") or {}
                check("retraining finished without error", st.get("last_error") is None, str(st.get("last_error"))[:300])
                g = res.get("gate_test", {})
                check("gate decided", "published" in res,
                      f"published={res.get('published')}, delta={g.get('delta', 0):+.4f}, "
                      f"95% CI [{g.get('ci_low', 0):+.4f}, {g.get('ci_high', 0):+.4f}]")
                print("      serving now:", call("GET", "/health")[1].get("model_version"))
                break

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
