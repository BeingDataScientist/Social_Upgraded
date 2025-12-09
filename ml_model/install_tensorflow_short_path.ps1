# Alternative: Install TensorFlow using a shorter virtual environment path
# This creates a venv in a shorter location and installs packages there

Write-Host "=== Installing TensorFlow with Short Path Workaround ===" -ForegroundColor Cyan
Write-Host ""

# Create a shorter path for venv
$shortVenvPath = "C:\venv_social_dec"

Write-Host "Step 1: Creating virtual environment in shorter path..." -ForegroundColor Yellow
Write-Host "Location: $shortVenvPath" -ForegroundColor Cyan

if (Test-Path $shortVenvPath) {
    Write-Host "Short venv path already exists. Removing..." -ForegroundColor Yellow
    Remove-Item -Path $shortVenvPath -Recurse -Force -ErrorAction SilentlyContinue
}

python -m venv $shortVenvPath

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "Virtual environment created successfully!" -ForegroundColor Green

# Activate the new venv
Write-Host "`nStep 2: Activating new virtual environment..." -ForegroundColor Yellow
& "$shortVenvPath\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`nStep 3: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel --no-cache-dir

# Install TensorFlow
Write-Host "`nStep 4: Installing TensorFlow..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Cyan

# Install dependencies first
pip install --no-cache-dir "numpy>=1.22,<1.25"
pip install --no-cache-dir packaging "protobuf<5.0.0"

# Install TensorFlow CPU (smaller)
pip install --no-cache-dir tensorflow-cpu==2.13.0

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== SUCCESS! ===" -ForegroundColor Green
    
    # Verify
    python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nInstalling remaining requirements..." -ForegroundColor Yellow
        pip install --no-cache-dir -r requirements.txt
        
        Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
        Write-Host "`nIMPORTANT: To use this environment, activate it with:" -ForegroundColor Yellow
        Write-Host "& '$shortVenvPath\Scripts\Activate.ps1'" -ForegroundColor Cyan
        Write-Host "`nOr update your scripts to use this path." -ForegroundColor Yellow
    }
} else {
    Write-Host "`n=== Installation failed ===" -ForegroundColor Red
    Write-Host "Please try enabling Windows long path support (see INSTALL_FIX.md)" -ForegroundColor Yellow
    exit 1
}

