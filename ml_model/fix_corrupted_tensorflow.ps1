# Fix Corrupted TensorFlow Installation
# This script removes the broken TensorFlow and reinstalls it properly

Write-Host "=== Fixing Corrupted TensorFlow Installation ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Uninstall broken TensorFlow
Write-Host "Step 1: Uninstalling corrupted TensorFlow..." -ForegroundColor Yellow
pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-intel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Some TensorFlow packages may not have been installed" -ForegroundColor Yellow
}

# Step 2: Clean up any remaining TensorFlow files manually
Write-Host "`nStep 2: Cleaning up TensorFlow cache and files..." -ForegroundColor Yellow
pip cache purge

# Step 3: Remove TensorFlow directories if they exist (may fail due to path length, that's OK)
Write-Host "`nStep 3: Attempting to remove TensorFlow directories..." -ForegroundColor Yellow
$tensorflowPath = "venv\Lib\site-packages\tensorflow"
if (Test-Path $tensorflowPath) {
    try {
        Remove-Item -Path $tensorflowPath -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Removed TensorFlow directory" -ForegroundColor Green
    } catch {
        Write-Host "Could not remove TensorFlow directory (path too long), continuing..." -ForegroundColor Yellow
    }
}

# Step 4: Upgrade pip tools
Write-Host "`nStep 4: Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to upgrade pip tools" -ForegroundColor Red
    exit 1
}

# Step 5: Install dependencies first
Write-Host "`nStep 5: Installing TensorFlow dependencies..." -ForegroundColor Yellow
pip install --no-cache-dir "numpy>=1.23.0,<2.0.0"
pip install --no-cache-dir packaging
pip install --no-cache-dir opt-einsum
pip install --no-cache-dir setuptools
pip install --no-cache-dir protobuf

# Step 6: Install TensorFlow
Write-Host "`nStep 6: Installing TensorFlow (this may take 5-10 minutes)..." -ForegroundColor Yellow
Write-Host "Using --no-cache-dir to avoid path length issues..." -ForegroundColor Cyan

# Try installing TensorFlow with specific version that works better on Windows
pip install --no-cache-dir "tensorflow>=2.12.0,<2.16.0"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nStandard TensorFlow installation failed, trying CPU-only version..." -ForegroundColor Yellow
    pip install --no-cache-dir "tensorflow-cpu>=2.12.0,<2.16.0"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n=== Installation failed ===" -ForegroundColor Red
        Write-Host "Trying older stable version..." -ForegroundColor Yellow
        pip install --no-cache-dir tensorflow==2.13.0
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`n=== All installation attempts failed ===" -ForegroundColor Red
            Write-Host "Please try:" -ForegroundColor Yellow
            Write-Host "1. Enable Windows long path support (see INSTALL_FIX.md)" -ForegroundColor Yellow
            Write-Host "2. Move project to a shorter path" -ForegroundColor Yellow
            Write-Host "3. Use Conda instead of pip" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Step 7: Verify installation
Write-Host "`nStep 7: Verifying TensorFlow installation..." -ForegroundColor Yellow
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print('Installation successful!')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== SUCCESS! TensorFlow is now properly installed ===" -ForegroundColor Green
    
    # Install remaining requirements
    Write-Host "`nInstalling remaining requirements..." -ForegroundColor Yellow
    pip install --no-cache-dir -r requirements.txt
    
    Write-Host "`n=== All Done! ===" -ForegroundColor Green
    Write-Host "You can now run: python train_models.py" -ForegroundColor Green
} else {
    Write-Host "`n=== Verification failed ===" -ForegroundColor Red
    Write-Host "TensorFlow may still be corrupted. Try:" -ForegroundColor Yellow
    Write-Host "1. Recreate virtual environment" -ForegroundColor Yellow
    Write-Host "2. Use a shorter project path" -ForegroundColor Yellow
    exit 1
}

