Write-Host "🌊 Initializing Flood Risk Prediction System via Docker..." -ForegroundColor Cyan

# Check for .env file
if (-not (Test-Path ".env")) {
    Write-Warning ".env file not found! Please create it before running."
    exit
}

Write-Host "Building and starting containers... (This may take a while for the first run)" -ForegroundColor Yellow
docker compose -f docker/docker-compose.yml up --build -d

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build/start failed! Please check your internet connection and Docker status."
    exit
}

Write-Host "`nWaiting for API to be healthy..." -ForegroundColor Yellow
while ($(docker inspect --format='{{json .State.Health.Status}}' flood-api 2>$null) -ne '"healthy"') {
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 2
    
    # Check if container is still running
    if (-not (docker ps -q -f name=flood-api)) {
        Write-Error "`nContainer 'flood-api' failed to start or crashed."
        exit
    }
}

Write-Host "`n`n✅ System is UP and RUNNING!" -ForegroundColor Green
Write-Host "------------------------------------"
Write-Host "API:       http://localhost:8000"
Write-Host "Dashboard: http://localhost:8501"
Write-Host "Docs:      http://localhost:8000/docs"
Write-Host "------------------------------------"
Write-Host "To view logs, run: docker compose -f docker/docker-compose.yml logs -f"

