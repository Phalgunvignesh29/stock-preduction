# Start the FastAPI Backend
Write-Host "Starting Backend FastAPI server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; uvicorn api:app --host 127.0.0.1 --port 8006 --reload"

# Start the Next.js Frontend
Write-Host "Starting Frontend Next.js server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; pnpm dev"

Write-Host "Waiting a few seconds for servers to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Open browser
Write-Host "Opening Application in browser..." -ForegroundColor Green
Start-Process "http://localhost:3000"
