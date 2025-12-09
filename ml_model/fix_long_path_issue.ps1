# Comprehensive Fix for Windows Long Path Issue with TensorFlow
# This script uses multiple strategies to work around Windows MAX_PATH limitation

Write-Host "=== Fixing Windows Long Path Issue for TensorFlow ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin (needed for some fixes)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Step 1: Upgrade pip first (critical - old pip doesn't handle long paths well)
Write-Host "Step 1: Upgrading pip (old version doesn't handle long paths)..." -ForegroundColor Yellow
python -m pip install --upgrade pip --no-cache-dir
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
}

# Step 2: Check if long paths are enabled
Write-Host "`nStep 2: Checking Windows long path support..." -ForegroundColor Yellow
$longPathEnabled = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -ErrorAction SilentlyContinue

if ($null -eq $longPathEnabled -or $longPathEnabled.LongPathsEnabled -eq 0) {
    Write-Host "Long path support is DISABLED" -ForegroundColor Red
    if ($isAdmin) {
        Write-Host "Attempting to enable long path support (requires restart)..." -ForegroundColor Yellow
        try {
            New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force | Out-Null
            Write-Host "Long path support enabled! You MUST restart your computer for this to take effect." -ForegroundColor Green
            Write-Host "After restart, run this script again." -ForegroundColor Yellow
            $restart = Read-Host "Restart now? (y/n)"
            if ($restart -eq 'y') {
                Restart-Computer
            }
        } catch {
            Write-Host "Failed to enable long path support: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "To enable long path support, run PowerShell as Administrator and execute:" -ForegroundColor Yellow
        Write-Host "New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force" -ForegroundColor Cyan
        Write-Host "Then restart your computer." -ForegroundColor Yellow
    }
} else {
    Write-Host "Long path support is ENABLED" -ForegroundColor Green
}

# Step 3: Create a junction/symlink to shorten the path
Write-Host "`nStep 3: Creating shorter path using junction..." -ForegroundColor Yellow
$shortPath = "C:\Social_Dec"
$currentPath = (Get-Location).Path

if (-not (Test-Path $shortPath)) {
    if ($isAdmin) {
        try {
            New-Item -ItemType Directory -Path $shortPath -Force | Out-Null
            cmd /c mklink /J "$shortPath\ml_model" "$currentPath" 2>$null
            Write-Host "Created junction at: $shortPath\ml_model" -ForegroundColor Green
            Write-Host "You can use this shorter path for installation." -ForegroundColor Yellow
        } catch {
            Write-Host "Could not create junction (may need admin): $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "To create a shorter path, run as Administrator" -ForegroundColor Yellow
    }
}

# Step 4: Clean up any partial TensorFlow installation
Write-Host "`nStep 4: Cleaning up partial TensorFlow installation..." -ForegroundColor Yellow
pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-intel 2>$null
pip cache purge

# Step 5: Set environment variables to help with path issues
Write-Host "`nStep 5: Setting environment variables..." -ForegroundColor Yellow
$env:TMPDIR = $env:TEMP
$env:TMP = $env:TEMP
$env:PIP_NO_CACHE_DIR = "1"

# Step 6: Try installing TensorFlow CPU-only (smaller, fewer files)
Write-Host "`nStep 6: Installing TensorFlow CPU-only (smaller package)..." -ForegroundColor Yellow
Write-Host "This version has fewer files and may avoid path length issues..." -ForegroundColor Cyan

# First install dependencies
pip install --no-cache-dir "numpy>=1.22,<1.25" --upgrade
pip install --no-cache-dir packaging protobuf "protobuf<5.0.0"

# Try TensorFlow CPU
pip install --no-cache-dir tensorflow-cpu==2.13.0

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== SUCCESS! TensorFlow CPU installed ===" -ForegroundColor Green
    
    # Verify
    Write-Host "`nVerifying installation..." -ForegroundColor Yellow
    python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print('Installation successful!')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nInstalling remaining requirements..." -ForegroundColor Yellow
        pip install --no-cache-dir -r requirements.txt
        Write-Host "`n=== All Done! ===" -ForegroundColor Green
    }
} else {
    Write-Host "`n=== Installation failed ===" -ForegroundColor Red
    Write-Host "`nRECOMMENDED SOLUTIONS:" -ForegroundColor Yellow
    Write-Host "1. Enable Windows long path support (see above) and restart" -ForegroundColor Cyan
    Write-Host "2. Move project to shorter path (e.g., C:\Projects\Social_Dec)" -ForegroundColor Cyan
    Write-Host "3. Use Conda instead of pip (handles paths better)" -ForegroundColor Cyan
    Write-Host "4. Create venv in shorter location and use --target flag" -ForegroundColor Cyan
    exit 1
}

