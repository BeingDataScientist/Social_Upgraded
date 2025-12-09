# Fix TensorFlow Installation on Windows
# This script addresses the Windows MAX_PATH limitation issue

Write-Host "=== TensorFlow Installation Fix for Windows ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clear pip cache
Write-Host "Step 1: Clearing pip cache..." -ForegroundColor Yellow
pip cache purge
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: pip cache purge failed, continuing anyway..." -ForegroundColor Yellow
}

# Step 2: Upgrade pip, setuptools, and wheel
Write-Host "`nStep 2: Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to upgrade pip tools" -ForegroundColor Red
    exit 1
}

# Step 3: Set environment variable to use shorter paths
Write-Host "`nStep 3: Setting environment variables for shorter paths..." -ForegroundColor Yellow
$env:TMPDIR = $env:TEMP
$env:TMP = $env:TEMP

# Step 4: Install TensorFlow with specific options to avoid path issues
Write-Host "`nStep 4: Installing TensorFlow with workarounds..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow

# Use --no-cache-dir and --no-deps to avoid path length issues
# First install dependencies separately
Write-Host "Installing TensorFlow dependencies..." -ForegroundColor Cyan
pip install --no-cache-dir numpy>=1.23.0
pip install --no-cache-dir packaging
pip install --no-cache-dir opt-einsum
pip install --no-cache-dir setuptools

# Install TensorFlow
Write-Host "`nInstalling TensorFlow (this may take 5-10 minutes)..." -ForegroundColor Cyan
pip install --no-cache-dir --upgrade tensorflow>=2.12.0

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== SUCCESS! TensorFlow installed successfully ===" -ForegroundColor Green
    Write-Host "`nInstalling remaining requirements..." -ForegroundColor Yellow
    pip install --no-cache-dir -r requirements.txt
} else {
    Write-Host "`n=== Installation failed ===" -ForegroundColor Red
    Write-Host "Trying alternative method..." -ForegroundColor Yellow
    
    # Alternative: Install TensorFlow CPU-only version (smaller, fewer dependencies)
    Write-Host "Attempting to install TensorFlow CPU-only version..." -ForegroundColor Cyan
    pip install --no-cache-dir tensorflow-cpu>=2.12.0
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n=== SUCCESS! TensorFlow CPU installed ===" -ForegroundColor Green
        pip install --no-cache-dir -r requirements.txt
    } else {
        Write-Host "`n=== Both methods failed ===" -ForegroundColor Red
        Write-Host "Please see the manual solutions in the README" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
Write-Host "You can now run your training script." -ForegroundColor Green

