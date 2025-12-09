@echo off
REM Fix Corrupted TensorFlow Installation
REM This script removes the broken TensorFlow and reinstalls it properly

echo === Fixing Corrupted TensorFlow Installation ===
echo.

REM Step 1: Uninstall broken TensorFlow
echo Step 1: Uninstalling corrupted TensorFlow...
python -m pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-intel
if errorlevel 1 (
    echo Warning: Some TensorFlow packages may not have been installed
)

REM Step 2: Clean up cache
echo.
echo Step 2: Cleaning up pip cache...
python -m pip cache purge

REM Step 3: Upgrade pip tools
echo.
echo Step 3: Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo Error: Failed to upgrade pip tools
    exit /b 1
)

REM Step 4: Install dependencies first
echo.
echo Step 4: Installing TensorFlow dependencies...
python -m pip install --no-cache-dir "numpy>=1.23.0,<2.0.0"
python -m pip install --no-cache-dir packaging
python -m pip install --no-cache-dir opt-einsum
python -m pip install --no-cache-dir setuptools
python -m pip install --no-cache-dir protobuf

REM Step 5: Install TensorFlow
echo.
echo Step 5: Installing TensorFlow (this may take 5-10 minutes)...
echo Using --no-cache-dir to avoid path length issues...
python -m pip install --no-cache-dir "tensorflow>=2.12.0,<2.16.0"

if errorlevel 1 (
    echo.
    echo Standard TensorFlow installation failed, trying CPU-only version...
    python -m pip install --no-cache-dir "tensorflow-cpu>=2.12.0,<2.16.0"
    
    if errorlevel 1 (
        echo.
        echo Installation failed, trying older stable version...
        python -m pip install --no-cache-dir tensorflow==2.13.0
        
        if errorlevel 1 (
            echo.
            echo === All installation attempts failed ===
            echo Please see INSTALL_FIX.md for alternative solutions
            exit /b 1
        )
    )
)

REM Step 6: Verify installation
echo.
echo Step 6: Verifying TensorFlow installation...
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print('Installation successful!')"

if errorlevel 1 (
    echo.
    echo === Verification failed ===
    echo TensorFlow may still be corrupted. Try recreating virtual environment.
    exit /b 1
) else (
    echo.
    echo === SUCCESS! TensorFlow is now properly installed ===
    echo.
    echo Installing remaining requirements...
    python -m pip install --no-cache-dir -r requirements.txt
    echo.
    echo === All Done! ===
    echo You can now run: python train_models.py
)

pause

