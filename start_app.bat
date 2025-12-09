@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  Digital Media ^& Mental Health Assessment
echo  Docker Container Management Script
echo ========================================
echo.

REM Check if argument is provided
if "%1"=="stop" goto :stop
if "%1"=="down" goto :stop
if "%1"=="start" goto :start
if "%1"=="restart" goto :restart

REM Show menu if no argument
:menu
echo Select an option:
echo   1. Start Application
echo   2. Stop Application
echo   3. Restart Application
echo   4. View Logs
echo   5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto :start
if "%choice%"=="2" goto :stop
if "%choice%"=="3" goto :restart
if "%choice%"=="4" goto :logs
if "%choice%"=="5" exit /b 0
echo Invalid choice. Please try again.
echo.
goto :menu

:start
echo ========================================
echo  Starting Application
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
if %errorlevel% neq 0 (
    echo ERROR: Docker build failed!
    pause
    exit /b 1
)

echo [3/4] Starting the application container...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start container!
    pause
    exit /b 1
)

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
echo To stop the application, run: start_app.bat stop
echo To view logs, run: start_app.bat logs
echo.
pause
exit /b 0

:stop
echo ========================================
echo  Stopping Application
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed!
    pause
    exit /b 1
)

echo Stopping Docker containers...
docker-compose down

echo.
echo Application stopped successfully!
echo.
pause
exit /b 0

:restart
echo ========================================
echo  Restarting Application
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed!
    pause
    exit /b 1
)

echo Stopping existing containers...
docker-compose down

echo.
echo Starting application...
goto :start

:logs
echo ========================================
echo  Viewing Application Logs
echo ========================================
echo.
echo Press Ctrl+C to exit log view
echo.

docker-compose logs -f

pause
exit /b 0
