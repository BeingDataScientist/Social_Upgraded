@echo off
REM Fix TensorFlow Installation on Windows
REM This script addresses the Windows MAX_PATH limitation issue

echo === TensorFlow Installation Fix for Windows ===
echo.

REM Step 1: Clear pip cache
echo Step 1: Clearing pip cache...
python -m pip cache purge
if errorlevel 1 (
    echo Warning: pip cache purge failed, continuing anyway...
)

REM Step 2: Upgrade pip, setuptools, and wheel
echo.
echo Step 2: Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo Error: Failed to upgrade pip tools
    exit /b 1
)

REM Step 3: Set environment variable to use shorter paths
echo.
echo Step 3: Setting environment variables for shorter paths...
set TMPDIR=%TEMP%
set TMP=%TEMP%

REM Step 4: Install TensorFlow with specific options to avoid path issues
echo.
echo Step 4: Installing TensorFlow with workarounds...
echo This may take several minutes...

REM Use --no-cache-dir to avoid path length issues
echo Installing TensorFlow dependencies...
python -m pip install --no-cache-dir numpy>=1.23.0
python -m pip install --no-cache-dir packaging
python -m pip install --no-cache-dir opt-einsum
python -m pip install --no-cache-dir setuptools

REM Install TensorFlow
echo.
echo Installing TensorFlow (this may take 5-10 minutes)...
python -m pip install --no-cache-dir --upgrade tensorflow>=2.12.0

if errorlevel 1 (
    echo.
    echo === Installation failed, trying alternative method ===
    echo Attempting to install TensorFlow CPU-only version...
    python -m pip install --no-cache-dir tensorflow-cpu>=2.12.0
    
    if errorlevel 1 (
        echo.
        echo === Both methods failed ===
        echo Please see the manual solutions in the README
        exit /b 1
    ) else (
        echo.
        echo === SUCCESS! TensorFlow CPU installed ===
    )
) else (
    echo.
    echo === SUCCESS! TensorFlow installed successfully ===
)

echo.
echo Installing remaining requirements...
python -m pip install --no-cache-dir -r requirements.txt

echo.
echo === Installation Complete ===
echo You can now run your training script.
pause

