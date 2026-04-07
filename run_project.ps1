Write-Host "Starting Flood Risk Prediction System..."
Start-Process "powershell" -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
Start-Process "powershell" -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8501"
