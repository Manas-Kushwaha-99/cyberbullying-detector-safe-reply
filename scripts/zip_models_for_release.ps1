# Cyberbullying Detector - Zip Models for GitHub Release
# Usage: Right-click -> "Run with PowerShell" or:
#        PowerShell -ExecutionPolicy Bypass -File scripts\zip_models_for_release.ps1

$ErrorActionPreference = "Stop"

$ModelsDir = "models(New)"
$ReleaseDir = "release_assets"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Zipping Models for GitHub Release" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Verify models exist
$required = @("distilbert", "distilbert_lora", "flan_t5_small_reply", "lstm.pt", "lstm_vocab.pkl", "tfidf_svm.pkl")
$missing = @()
foreach ($item in $required) {
    $path = Join-Path $ModelsDir $item
    if (-not (Test-Path $path)) {
        $missing += $item
    }
}

if ($missing.Count -gt 0) {
    Write-Host "[ERROR] Missing models in $ModelsDir\:" -ForegroundColor Red
    foreach ($m in $missing) {
        Write-Host "        - $m" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Please train all models before creating release assets." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create release directory
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

# Zip detection models
Write-Host "[1/2] Zipping detection models..." -ForegroundColor Green
$detectionItems = @(
    "distilbert",
    "distilbert_lora",
    "lstm.pt",
    "lstm_vocab.pkl",
    "tfidf_svm.pkl"
)

$detectionZip = "$ReleaseDir\detection_models.zip"
if (Test-Path $detectionZip) { Remove-Item $detectionZip }

foreach ($item in $detectionItems) {
    $source = Join-Path $ModelsDir $item
    $dest = Join-Path (Join-Path $ReleaseDir "temp_detection") $item
    if (Test-Path $source -PathType Container) {
        Copy-Item -Recurse -Force $source $dest
    } else {
        Copy-Item -Force $source $dest
    }
}

Compress-Archive -Path "$ReleaseDir\temp_detection\*" -DestinationPath $detectionZip -Force
Remove-Item -Recurse -Force "$ReleaseDir\temp_detection"

$detectionSize = (Get-Item $detectionZip).Length / 1MB
$detectionSizeStr = "{0:N1}" -f $detectionSize
Write-Host "      Created: detection_models.zip ($detectionSizeStr MB)" -ForegroundColor Green

# Zip reply model
Write-Host "[2/2] Zipping reply generator model..." -ForegroundColor Green
$replyZip = "$ReleaseDir\reply_model.zip"
if (Test-Path $replyZip) { Remove-Item $replyZip }

$source = Join-Path $ModelsDir "flan_t5_small_reply"
    $dest = Join-Path (Join-Path $ReleaseDir "temp_reply") "flan_t5_small_reply"
Copy-Item -Recurse -Force $source $dest
Compress-Archive -Path "$ReleaseDir\temp_reply\*" -DestinationPath $replyZip -Force
Remove-Item -Recurse -Force "$ReleaseDir\temp_reply"

$replySize = (Get-Item $replyZip).Length / 1MB
$replySizeStr = "{0:N1}" -f $replySize
Write-Host "      Created: reply_model.zip ($replySizeStr MB)" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Release Assets Ready!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Upload these files to GitHub Releases:" -ForegroundColor Yellow
Write-Host "  1. $ReleaseDir\detection_models.zip" -ForegroundColor White
Write-Host "  2. $ReleaseDir\reply_model.zip" -ForegroundColor White
Write-Host ""
Write-Host "Update these values in scripts\download_models.ps1:" -ForegroundColor Yellow
Write-Host "  `$GitHubUser = \"YOUR_USERNAME\"" -ForegroundColor White
Write-Host "  `$GitHubRepo = \"YOUR_REPO\"" -ForegroundColor White
Write-Host "  `$ReleaseTag = \"v1.0.0\"" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"
