@echo off
REM Setup Virtual Environment in Root Folder
REM This script creates a venv in the root directory and installs all dependencies

echo ========================================
echo  Setting Up Virtual Environment
echo ========================================
echo.

REM Check if venv already exists
if exist "venv" (
    echo Virtual environment already exists at: venv
    set /p response="Do you want to recreate it? (y/n): "
    if /i "%response%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
    ) else (
        echo Using existing virtual environment.
        echo.
        echo To activate it, run:
        echo   venv\Scripts\activate.bat
        pause
        exit /b 0
    )
)

echo [1/3] Creating virtual environment in root folder...
python -m venv venv

if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    echo Make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

echo Virtual environment created successfully!

echo.
echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/3] Installing all dependencies from requirements.txt...
echo This may take a few minutes...

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install all requirements
pip install --no-cache-dir -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some dependencies!
    echo Please check the error messages above.
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo  SUCCESS! Virtual environment is ready!
    echo ========================================
    echo.
    echo To activate the virtual environment, run:
    echo   venv\Scripts\activate.bat
    echo.
    echo To run the application:
    echo   python app.py
    echo.
)

pause

