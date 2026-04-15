# 🌊 Flood Risk Prediction System — Local Launcher
Write-Host "Checking environment configurations..." -ForegroundColor Cyan

# 1. Check for .env file
if (-not (Test-Path ".env")) {
    Write-Error "ERROR: .env file not found. Please copy .env.example to .env and fill in your API tokens."
    exit 1
}

# 2. Check for Trained Model
if (-not (Test-Path "models/checkpoints/best_lstm_model.pt")) {
    Write-Warning "WARNING: Trained LSTM model not found at models/checkpoints/best_lstm_model.pt"
    Write-Warning "The system will start, but predictions will use synthetic fallbacks until you run the training script."
}

Write-Host "`nStarting API Service (Port 8000)..." -ForegroundColor Green
Start-Process "powershell" -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

Write-Host "Waiting for API to initialize (5s)..."
Start-Sleep -Seconds 5

Write-Host "Starting Analytics Dashboard (Port 8501)..." -ForegroundColor Green
Start-Process "powershell" -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8501"

Write-Host "`n🚀 System is now launching in separate windows!" -ForegroundColor Cyan
Write-Host "- API: http://localhost:8000"
Write-Host "- Dashboard: http://localhost:8501"
