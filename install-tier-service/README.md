# Install-tier prediction service

Model + backend for *Predicting Google Play App Install Tiers*: a Random Forest trained on
`googleplaystore_classification.csv` (60 Phase 2 features, no preprocessing repeated), served by a
FastAPI API, with user feedback and automatic retraining. Sized to run on a laptop with ~6 GB of RAM.

| File | What it does |
|---|---|
| `colab_train.ipynb` | Trains the real model in Colab (step 1) |
| `train.py` | Training: reads the processed CSV like the Milestone 3 notebook, builds the developer table from the raw Kaggle CSV, checks parity (PASSED/FAILED), fits the forest, publishes `rf-v1` |
| `server.py` | The API, the SQLite feedback store and the retraining job (`python server.py retrain` / `rollback`) |
| `common.py` | Shared code: settings, Phase 2 feature pipeline, statistics, metrics, model registry |
| `check_service.py` | Checks a running server end to end, prints PASS/FAIL |
| `test_service.py` | 24 automated tests on synthetic data (no model needed) |
| `api_examples.json` | Real request/response examples for the web client |
| `requirements.txt` | Packages (scikit-learn pinned: train and serve with the same version) |

Created by training: `registry/` (model versions, `current.json`, `history.jsonl`) and `data/`
(`train_X.npy`, `train_y.npy`, `test_X.npy`, `test_y.npy` for retraining; `feedback.db`).

## Model size

The report setting (100 trees, `min_samples_leaf` 5) scores macro-F1 0.4929 but needs ~5 GB of RAM, more
than a 6 GB laptop has. Colab training therefore runs a model search (`train.py --search`): it tries several
`trees x min_samples_leaf` combinations, scores each on a validation split taken from the training data, and
keeps the highest macro-F1 whose model file fits `--max-model-gb` (0.4 GB by default). With a model of that
size the server needs ~0.7 GB and retraining ~1.6 GB more (measured on full-size data).

---

## Step 1 — Train in Colab (once, ~30-40 minutes)

1. In Google Drive, open `MyDrive/ML-Datasets/` (it holds `googleplaystore_classification.csv`).
   **Delete any older `install-tier-service` folder or zip**, then upload this whole
   `install-tier-service` folder (or the zip).
2. Double-click `install-tier-service/colab_train.ipynb` → Open with Google Colaboratory.
3. Runtime → **Run all**. Allow Drive access when asked.
4. Check the output, cell by cell:
   - `Code version: 2026-11-27 · 60 features, with developer frequency`
   - `Raw CSV: …/Google-Playstore.csv`
   - `Loaded 2,312,944 apps x 60 features`
   - `PASSED: the server builds the same features as Phase 2.`
   - the search table: one line per candidate with validation macro-F1, model size and recall per tier
   - `Chosen: … trees, min_samples_leaf …`
   - `Test macro-F1 …`, `Test recall`, `Test precision` — copy these and the table into the report
     (the report setting gave 0.4929)
   - `Saved registry/models/rf-v1.joblib (… MB …)` — at most ~430 MB
5. Download `MyDrive/ML-Datasets/model_outputs.zip` to your PC.

To try other combinations, edit `CANDIDATES` in the Train cell (format `trees x leaf`, comma separated) and
run that cell and the last cell again. Raise `MAX_MODEL_GB` only if the PC running the server has more RAM.

## Step 2 — Set up on your PC (once)

Open **Command Prompt** (cmd):

```bat
cd /d F:\install-tier-service\install-tier-service
python -m pip install -r requirements.txt
```

Delete the old `registry` and `data` folders in this folder, then extract `model_outputs.zip` here so that
`registry` and `data` sit next to `server.py`. Check:

```bat
dir registry\models data
```

You should see `rf-v1.joblib` (a few hundred MB) and the four `.npy` files.

## Step 3 — Automated tests (no server needed)

```bat
python -m pytest -q
```

Expect `24 passed`.

## Step 4 — Start the server

```bat
set ADMIN_TOKEN=secret
python -m uvicorn server:app --port 8000
```

Wait for `Serving rf-v1 (loaded in N s)` (a few seconds). Keep this window open; Ctrl+C stops the server.
Open http://localhost:8000/docs to try the endpoints in the browser.

## Step 5 — Check everything (second cmd window)

```bat
cd /d F:\install-tier-service\install-tier-service
python check_service.py
```

Expect all checks PASS (service, predictions, edge inputs, validation) and three sample profiles.

Then the online-learning loop. To keep test reports out of the real feedback, restart the server
first with a test database (in the server window: Ctrl+C, then):

```bat
set DB_PATH=data\check_feedback.db
python -m uvicorn server:app --port 8000
```

and in the second window:

```bat
python check_service.py --feedback --retrain --token secret
```

Retraining takes about 5–10 minutes. Expect `published=True` or `False` with the gate numbers
(Δ macro-F1 and its 95% CI), then `serving now: rf-v2` if it was published, and `N/N checks passed`.

Afterwards, go back to the original model: stop the server and run

```bat
python server.py rollback
```

then open a **new** cmd window (so `DB_PATH` is not set) and start the server as in Step 4.

## Step 6 — Deploy (run it for real / for the demo)

```bat
cd /d F:\install-tier-service\install-tier-service
set ADMIN_TOKEN=choose-a-secret
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

- The web client calls `http://localhost:8000` (or `http://<this PC's IP>:8000` from another device on
  the same network; allow Python through the Windows firewall when asked).
- Real feedback is stored in `data\feedback.db`.
- Retraining runs by itself every Sunday 03:00 UTC, or after 1,000 new reports. For the demo, start it
  with `POST /admin/retrain` (header `X-Admin-Token`) and watch `GET /admin/retrain/status`.
- A new model is published only if it is not significantly worse on the fixed test set; the server switches
  to it within 30 seconds without restarting. `python server.py rollback` undoes it.
- Close memory-heavy programs (many browser tabs) while retraining runs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` returns 503 | The model is still loading; wait for `Serving rf-v1` |
| `No module named …` | `python -m pip install -r requirements.txt` |
| `The filename, directory name, or volume label syntax is incorrect` | You typed a PowerShell command in cmd; use `set NAME=value` |
| `NameError: name 'true' is not defined` in server.py | server.py was overwritten with JSON; extract it again from the zip |
| Server stops answering while retraining | Not enough free RAM: close programs, or retrain with a larger `MIN_LEAF` |
