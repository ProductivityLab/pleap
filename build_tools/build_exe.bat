@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

echo ============================================================
echo  PLEAP v1.1.0 - Build standalone EXE
echo ============================================================
echo Working dir: %CD%
echo.

REM --- detect non-ASCII / spaces in path (common PyInstaller TOC cause) ---
echo %CD% | findstr /R "[^a-zA-Z0-9_\\:.\-]" >nul
if not errorlevel 1 (
    echo [WARN] Current path may contain spaces or non-ASCII characters.
    echo This often causes PyInstaller TOC/path errors.
    echo Consider moving this folder to e.g.  C:\PLEAP\  and retrying.
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 ( pause & exit /b 1 )
)

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    pause & exit /b 1
)

if not exist venv (
    echo [1/6] Creating venv...
    python -m venv venv
    if errorlevel 1 ( echo venv creation failed & pause & exit /b 1 )
)

echo [2/6] Activating venv...
call venv\Scripts\activate.bat

echo [3/6] Installing dependencies...
python -m pip install --upgrade pip wheel setuptools >nul
pip install -e ".[gui,build]"
if errorlevel 1 ( echo pip install failed & pause & exit /b 1 )

echo [4/6] Verifying imports...
python -c "import PyQt5.QtCore, numpy, pandas, scipy, statsmodels, openpyxl, matplotlib; print('OK')"
if errorlevel 1 ( echo [ERROR] Required modules missing & pause & exit /b 1 )

echo [5/6] Cleaning previous build artifacts (fixes 'toc not found' errors)...
if exist build       rmdir /s /q build       2>nul
if exist dist        rmdir /s /q dist        2>nul
if exist __pycache__ rmdir /s /q __pycache__ 2>nul
if exist PLEAP.spec  del /q     PLEAP.spec   2>nul
REM PyInstaller global cache (TOC files live here)
if exist "%LOCALAPPDATA%\pyinstaller" rmdir /s /q "%LOCALAPPDATA%\pyinstaller" 2>nul
for /d %%D in ("%TEMP%\_MEI*") do rmdir /s /q "%%D" 2>nul

echo [6/6] Running PyInstaller...
pyinstaller --noconfirm --clean ^
    --name PLEAP ^
    --windowed ^
    --icon src\pleap\assets\lab_logo.ico ^
    --workpath "%CD%\build" ^
    --distpath "%CD%\dist" ^
    --specpath "%CD%" ^
    --add-data "src\pleap\assets;pleap\assets" ^
    --add-data "examples;examples" ^
    --paths src ^
    --collect-all PyQt5 ^
    --collect-submodules statsmodels ^
    --collect-submodules scipy ^
    --collect-submodules matplotlib ^
    --hidden-import matplotlib.backends.backend_qt5agg ^
    --hidden-import PyQt5.sip ^
    --hidden-import openpyxl ^
    --hidden-import pandas ^
    src\pleap\__main__.py
set RC=%errorlevel%

if not "%RC%"=="0" (
    echo.
    echo ============================================================
    echo [ERROR] PyInstaller failed with errorlevel %RC%.
    echo.
    echo If you see a 'toc' file error:
    echo   1. Close any running PLEAP.exe and any Explorer window on dist\
    echo   2. Add this folder to Windows Defender / antivirus exclusions
    echo   3. Move this project to a short ASCII path like C:\PLEAP\
    echo   4. Try Python 3.11 instead of 3.12 / 3.13
    echo ============================================================
    pause & exit /b %RC%
)

if not exist dist\PLEAP\PLEAP.exe (
    echo [ERROR] PyInstaller reported success but PLEAP.exe is missing.
    echo Likely cause: antivirus quarantined the output. Whitelist this folder.
    pause & exit /b 1
)

echo.
echo  OK -- dist\PLEAP\PLEAP.exe built successfully.
endlocal
exit /b 0
