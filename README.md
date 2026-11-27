Download Datasets from here: https://drive.google.com/drive/folders/1zoM2EMOJiOii8NUT3Z7jdsDKZNMt2Px-?usp=drive_link 

## Run the install predictor on Windows

Install Python 3.14 and Node.js 22.12 or newer. Check out this branch and open two PowerShell terminals in the repository root.

### 1. Copy your trained model

Place your existing rf-v1.joblib at install-tier-service/registry/models/rf-v1.joblib. Create the models folder if needed. The checked-in registry/current.json selects rf-v1. Your local model file is ignored by Git.

### 2. Start the backend in the first terminal

Stop the mock server if it is running on port 8000.

```powershell
Set-Location .\install-tier-service
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:ENABLE_SCHEDULER = '0'
$env:DB_PATH = 'data/local-test-feedback.db'
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. Open [API health](http://127.0.0.1:8000/health) and confirm it shows status ready and model_version rf-v1.

### 3. Start the client in the second terminal

```powershell
Set-Location .\client
npm.cmd ci
$env:VITE_API_URL = 'http://127.0.0.1:8000'
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open the [predictor](http://127.0.0.1:5173/). Use the prefilled app details, check **Let me report actual installs later**, and click **Predict installs**. A tier, install range, and four confidence bars should appear. Enter a sample count such as 250000 and click **Submit feedback**. The form should say **Feedback saved**.

For automated API checks, open another terminal in the repository root and run:

```powershell
Set-Location .\install-tier-service
.\.venv\Scripts\python.exe check_service.py --url http://127.0.0.1:8000 --feedback
```

The expected result is 27/27 checks passed. Sample feedback stays in your local test database. For training or retraining setup, see the [backend README](install-tier-service/README.md).
