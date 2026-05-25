# Cyberbullying Detector - Model Downloader
# Downloads pre-trained models from GitHub Releases
# Usage: Right-click -> "Run with PowerShell" or:
#        PowerShell -ExecutionPolicy Bypass -File scripts\download_models.ps1

$ErrorActionPreference = "Stop"

# Configuration - UPDATE THESE WITH YOUR ACTUAL GITHUB INFO
$GitHubUser = "YOUR_USERNAME"
$GitHubRepo = "YOUR_REPO"
$ReleaseTag = "v1.0.0"

$BaseUrl = "https://github.com/$GitHubUser/$GitHubRepo/releases/download/$ReleaseTag"
$ModelsDir = "models_new"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Cyberbullying Detector Model Downloader" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if models already exist
$hasModels = Test-Path "$ModelsDir\distilbert_lora" -PathType Container
if ($hasModels) {
    Write-Host "Models already exist in $ModelsDir\" -ForegroundColor Yellow
    $confirm = Read-Host "Re-download? (y/N)"
    if ($confirm -ne 'y' -and $confirm -ne 'Y') {
        Write-Host "Cancelled."
        exit 0
    }
}

# Create temp directory
$TempDir = "temp_downloads"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

try {
    # Download detection models
    Write-Host "[1/2] Downloading detection models..." -ForegroundColor Green
    $DetectionUrl = "$BaseUrl/detection_models.zip"
    $DetectionZip = "$TempDir\detection_models.zip"
    
    try {
        Invoke-WebRequest -Uri $DetectionUrl -OutFile $DetectionZip -MaximumRedirection 5
        Write-Host "      Expanding detection_models.zip..." -ForegroundColor Gray
        Expand-Archive -Path $DetectionZip -DestinationPath $ModelsDir -Force
        Write-Host "      [OK] Detection models installed." -ForegroundColor Green
    } catch {
        Write-Host "      [FAIL] Could not download detection models." -ForegroundColor Red
        Write-Host "      Error: $_" -ForegroundColor Red
        Write-Host "      Please download manually from:" -ForegroundColor Yellow
        Write-Host "      $DetectionUrl" -ForegroundColor Yellow
    }
    
    # Download reply model
    Write-Host "[2/2] Downloading reply generator model..." -ForegroundColor Green
    $ReplyUrl = "$BaseUrl/reply_model.zip"
    $ReplyZip = "$TempDir\reply_model.zip"
    
    try {
        Invoke-WebRequest -Uri $ReplyUrl -OutFile $ReplyZip -MaximumRedirection 5
        Write-Host "      Expanding reply_model.zip..." -ForegroundColor Gray
        Expand-Archive -Path $ReplyZip -DestinationPath $ModelsDir -Force
        Write-Host "      [OK] Reply model installed." -ForegroundColor Green
    } catch {
        Write-Host "      [FAIL] Could not download reply model." -ForegroundColor Red
        Write-Host "      Error: $_" -ForegroundColor Red
        Write-Host "      Please download manually from:" -ForegroundColor Yellow
        Write-Host "      $ReplyUrl" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  Download Complete!" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "You can now run the app with: .\run.bat" -ForegroundColor Green
    
} finally {
    # Cleanup
    if (Test-Path $TempDir) {
        Remove-Item -Recurse -Force $TempDir
    }
}

Write-Host ""
Read-Host "Press Enter to exit"
