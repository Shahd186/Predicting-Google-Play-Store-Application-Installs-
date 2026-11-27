@echo off
rem Starts the model service and the web UI, then opens the browser.
start "Install-tier API (port 8000)" /D "%~dp0install-tier-service" cmd /k "set ADMIN_TOKEN=secret&& set ENABLE_SCHEDULER=0&& set DB_PATH=data\local-test-feedback.db&& .venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000"
start "Install-tier UI (port 5173)" /D "%~dp0client" cmd /k "set VITE_API_URL=http://127.0.0.1:8000&& npm run dev -- --host 127.0.0.1 --port 5173 --strictPort"
echo Waiting for the model to load...
timeout /t 12 /nobreak >nul
start "" http://127.0.0.1:5173/
