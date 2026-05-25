# Create Desktop Shortcut for Cyberbullying Detector
# Usage: Right-click -> "Run with PowerShell"

$ErrorActionPreference = "Stop"

$ProjectDir = (Get-Location).Path
$ShortcutName = "Cyberbullying Detector.lnk"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath $ShortcutName

Write-Host "Creating desktop shortcut..." -ForegroundColor Cyan

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "cmd.exe"
$Shortcut.Arguments = "/c `"$ProjectDir\run.bat`""
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.IconLocation = "%SystemRoot%\System32\SHELL32.dll,14"
$Shortcut.Description = "Cyberbullying Detector + Safe Reply Generator"
$Shortcut.Save()

Write-Host "[OK] Shortcut created on Desktop: $ShortcutName" -ForegroundColor Green
Write-Host "     Double-click it to launch the app!" -ForegroundColor Green

Read-Host "Press Enter to exit"
