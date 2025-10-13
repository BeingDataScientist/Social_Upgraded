@echo off
echo ========================================
echo  Digital Media & Mental Health Assessment
echo  Docker Container Startup Script
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [1/4] Checking if container is already running...
docker-compose ps | findstr "Up" >nul
if %errorlevel% equ 0 (
    echo Container is already running! Opening browser...
    goto :open_browser
)

echo [2/4] Building Docker image...
docker-compose build --no-cache

echo [3/4] Starting the application container...
docker-compose up -d

echo [4/4] Waiting for application to start...
timeout /t 10 /nobreak >nul

REM Check if container is running
docker-compose ps | findstr "Up" >nul
if %errorlevel% neq 0 (
    echo ERROR: Container failed to start!
    echo Checking logs...
    docker-compose logs
    pause
    exit /b 1
)

:open_browser
echo.
echo ========================================
echo  SUCCESS! Application is running!
echo ========================================
echo.
echo Application URL: http://localhost:5000
echo.
echo Opening browser...
start http://localhost:5000

echo.
echo Container Status:
docker-compose ps

echo.
echo To stop the application, run: docker-compose down
echo To view logs, run: docker-compose logs -f
echo.
pause
