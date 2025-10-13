Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Digital Media & Mental Health Assessment" -ForegroundColor Cyan
Write-Host " Docker Container Startup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not running"
    }
} catch {
    Write-Host "ERROR: Docker is not running or not installed!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/4] Checking if container is already running..." -ForegroundColor Yellow
$containerStatus = docker-compose ps
if ($containerStatus -match "Up") {
    Write-Host "Container is already running! Opening browser..." -ForegroundColor Green
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " SUCCESS! Application is running!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Application URL: http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "Opening browser..." -ForegroundColor Yellow
    Start-Process "http://localhost:5000"

    Write-Host ""
    Write-Host "Container Status:" -ForegroundColor Cyan
    docker-compose ps

    Write-Host ""
    Write-Host "To stop the application, run: docker-compose down" -ForegroundColor Yellow
    Write-Host "To view logs, run: docker-compose logs -f" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host "[2/4] Building Docker image..." -ForegroundColor Yellow
docker-compose build --no-cache

Write-Host "[3/4] Starting the application container..." -ForegroundColor Yellow
docker-compose up -d

Write-Host "[4/4] Waiting for application to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if container is running
$containerStatus = docker-compose ps
if ($containerStatus -notmatch "Up") {
    Write-Host "ERROR: Container failed to start!" -ForegroundColor Red
    Write-Host "Checking logs..." -ForegroundColor Yellow
    docker-compose logs
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " SUCCESS! Application is running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Application URL: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""

Write-Host "Opening browser..." -ForegroundColor Yellow
Start-Process "http://localhost:5000"

Write-Host ""
Write-Host "Container Status:" -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "To stop the application, run: docker-compose down" -ForegroundColor Yellow
Write-Host "To view logs, run: docker-compose logs -f" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"
