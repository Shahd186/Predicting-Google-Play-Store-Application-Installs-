# Run and test the install predictor on Windows

Shahd, use these steps to run Mariam’s React client with your FastAPI backend and trained Random Forest.

## 1. Get the branch and add your model

Install Python 3.14 and Node.js 22.12 or newer. On GitHub, select the **feature/web-client** branch and choose **Code → Download ZIP**. Extract it and open the folder containing both **client** and **install-tier-service**.

Copy your existing **rf-v1.joblib** to this path inside the extracted project. Create the **models** folder if needed.

    install-tier-service/registry/models/rf-v1.joblib

The included **registry/current.json** selects rf-v1.

## 2. Start FastAPI

If the mock server is running, stop it with **Ctrl+C** in its terminal. Open PowerShell in the extracted project folder and run:

```powershell
cd .\install-tier-service
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:ENABLE_SCHEDULER = '0'
$env:DB_PATH = 'data/local-test-feedback.db'
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Keep this window open. Wait for **Serving rf-v1**. Open [API health](http://127.0.0.1:8000/health) and check for:

```json
{"status":"ready","model_version":"rf-v1"}
```

## 3. Start the web app

Open a second PowerShell window in the extracted project folder and run:

```powershell
cd .\client
npm.cmd ci
$env:VITE_API_URL = 'http://127.0.0.1:8000'
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Keep both windows open and visit the [install predictor](http://127.0.0.1:5173/).

## 4. Test prediction and feedback

1. Use the prefilled app details, check **Let me report actual installs later**, and click **Predict installs**.
2. Check that a predicted tier, install range, four confidence bars, and two what-if comparisons appear.
3. Enter **250000** as a sample install count and click **Submit feedback**. The page should say **Feedback saved: 250,000 actual installs.** The sample is stored in your local test database.
4. To stop the app, press **Ctrl+C** in each PowerShell window.
