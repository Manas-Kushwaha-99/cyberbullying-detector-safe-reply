@echo off
setlocal EnableDelayedExpansion

:: ── CONFIGURATION ────────────────────────────────────────────────
set "REPO_OWNER=Manas-Kushwaha-99"
set "REPO_NAME=cyberbullying-detector-safe-reply"
set "RELEASE_TAG=v1.0.2"
set "BASE_URL=https://github.com/%REPO_OWNER%/%REPO_NAME%/releases/download/%RELEASE_TAG%"

:: ── CHECK IF ALREADY IN PROJECT ─────────────────────────────────
if exist "app.py" (
    set "PROJECT_DIR=%CD%"
    echo [INFO] Running setup from existing project folder: %CD%
    goto :SETUP_LOCAL
)

:: ── STANDALONE BOOTSTRAPPER ─────────────────────────────────────
echo ==========================================
echo  Cyberbullying Detector + Safe Reply
echo  One-Click Auto Setup
echo ==========================================
echo.

:: Check Python first
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo.
    echo Please install Python 3.9 or newer from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check "Add Python to PATH"
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

:: Create project folder
set "PROJECT_DIR=%CD%\Cyberbullying-Detector"
if exist "%PROJECT_DIR%" (
    echo [INFO] Project folder already exists: %PROJECT_DIR%
    echo          Setup will update / reuse existing files.
    echo.
)
if not exist "%PROJECT_DIR%" mkdir "%PROJECT_DIR%"

:: Download source code
echo [1/6] Downloading project source code...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/%REPO_OWNER%/%REPO_NAME%/archive/refs/tags/%RELEASE_TAG%.zip' -OutFile '%PROJECT_DIR%\source.zip' -MaximumRedirection 5 } catch { exit 1 }"
if errorlevel 1 (
    echo [ERROR] Failed to download source code.
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)

echo [1/6] Extracting source code...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%PROJECT_DIR%\source.zip' -DestinationPath '%PROJECT_DIR%\_src' -Force"
for /f "delims=" %%D in ('dir /b /ad "%PROJECT_DIR%\_src"') do (
    robocopy "%PROJECT_DIR%\_src\%%D" "%PROJECT_DIR%" /E /MOVE >nul 2>&1
)
if exist "%PROJECT_DIR%\_src" rmdir /S /Q "%PROJECT_DIR%\_src"
del "%PROJECT_DIR%\source.zip" >nul 2>&1
cd /d "%PROJECT_DIR%"

:SETUP_LOCAL
echo.
echo [2/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo          Virtual environment already exists.
)

echo [3/6] Installing dependencies (this may take 5-15 minutes)...
"%PROJECT_DIR%\venv\Scripts\python.exe" -m pip install --upgrade pip -q
"%PROJECT_DIR%\venv\Scripts\python.exe" -m pip install -r "%PROJECT_DIR%\requirements.txt" -q
if errorlevel 1 (
    echo [WARNING] Some packages may have failed, but continuing...
)

echo [4/6] Downloading NLTK data...
"%PROJECT_DIR%\venv\Scripts\python.exe" -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NLTK download may have failed, but app may still work.
)

echo [5/6] Downloading AI models...
if not exist "models_new" mkdir "models_new"

call :DOWNLOAD_DETECTION
call :DOWNLOAD_REPLY
goto :AFTER_DOWNLOADS

:: ── SUBROUTINES ──────────────────────────────────────────────────
:DOWNLOAD_DETECTION
if exist "models_new\distilbert_lora" goto :DETECTION_EXISTS
echo          Downloading detection_models.zip (~253 MB)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -Uri '%BASE_URL%/detection_models.zip' -OutFile 'detection_models.zip' -MaximumRedirection 5 } catch { exit 1 }"
if errorlevel 1 (
    echo [ERROR] Failed to download detection_models.zip.
    echo         You can manually download it from:
    echo         %BASE_URL%/detection_models.zip
    echo         and extract it to: %PROJECT_DIR%\models_new
    pause
    exit /b 1
)
echo          Extracting detection_models.zip...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Expand-Archive -Path 'detection_models.zip' -DestinationPath 'models_new' -Force"
del detection_models.zip
goto :EOF

:DETECTION_EXISTS
echo          Detection models already present.
goto :EOF

:DOWNLOAD_REPLY
if exist "models_new\flan_t5_small_reply" goto :REPLY_EXISTS
echo          Downloading reply_model.zip (~273 MB)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -Uri '%BASE_URL%/reply_model.zip' -OutFile 'reply_model.zip' -MaximumRedirection 5 } catch { exit 1 }"
if errorlevel 1 (
    echo [ERROR] Failed to download reply_model.zip.
    echo         You can manually download it from:
    echo         %BASE_URL%/reply_model.zip
    echo         and extract it to: %PROJECT_DIR%\models_new
    pause
    exit /b 1
)
echo          Extracting reply_model.zip...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Expand-Archive -Path 'reply_model.zip' -DestinationPath 'models_new' -Force"
del reply_model.zip
goto :EOF

:REPLY_EXISTS
echo          Reply model already present.
goto :EOF

:: ── AFTER DOWNLOADS ──────────────────────────────────────────────
:AFTER_DOWNLOADS

echo [6/6] Creating desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $sc = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Cyberbullying Detector.lnk'); $sc.TargetPath = '%PROJECT_DIR%\run.bat'; $sc.WorkingDirectory = '%PROJECT_DIR%'; $sc.IconLocation = '%SystemRoot%\System32\SHELL32.dll,14'; $sc.Description = 'Cyberbullying Detector + Safe Reply Generator'; $sc.Save()" >nul 2>&1

:: ── DONE ─────────────────────────────────────────────────────────
echo.
echo ==========================================
echo  Setup Complete!
echo ==========================================
echo.
echo Project installed at:
echo   %PROJECT_DIR%
echo.
echo A desktop shortcut has been created:
echo   Cyberbullying Detector
echo.
set /p launch="Launch the app now? [Y/n]: "
if /i "%launch%"=="" set launch=Y
if /i "%launch%"=="Y" (
    echo Launching app...
    start "" "%PROJECT_DIR%\run.bat"
) else (
    echo You can launch anytime by double-clicking the desktop shortcut.
)
echo.
pause
