@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

echo ============================================================
echo  PLEAP v1.12 - One-click installer build
echo  Working dir: %CD%
echo ============================================================
echo.

echo === Step 1/4: Build EXE ===
call build_tools\build_exe.bat
set RC=%errorlevel%
if not "%RC%"=="0" (
    echo.
    echo [ERROR] build_exe.bat returned errorlevel %RC%.
    echo See messages above for the cause.
    pause & exit /b %RC%
)
if not exist dist\PLEAP\PLEAP.exe (
    echo [ERROR] dist\PLEAP\PLEAP.exe was not produced.
    pause & exit /b 1
)
echo  OK: dist\PLEAP\PLEAP.exe exists.
echo.

echo === Step 2/4: Locate Inno Setup (ISCC.exe) ===
set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC (
    where iscc >nul 2>nul
    if not errorlevel 1 set "ISCC=iscc"
)
if not defined ISCC (
    echo [ERROR] Inno Setup 6 not found.
    echo Download Inno Setup 6.3+  https://jrsoftware.org/isdl.php
    pause & exit /b 1
)
echo  Using ISCC: %ISCC%
echo.

echo === Step 3/4: Ensure Output\ exists ===
if not exist Output mkdir Output
echo.

echo === Step 4/4: Compile installer ===
"%ISCC%" build_tools\installer.iss
set RC=%errorlevel%
if not "%RC%"=="0" (
    echo.
    echo [ERROR] Inno Setup returned errorlevel %RC%.
    echo Common causes:
    echo   - Inno Setup older than 6.3 (update from jrsoftware.org)
    echo   - LICENSE.txt or src\pleap\assets\lab_logo.ico missing
    echo   - Antivirus blocked file write to Output\
    pause & exit /b %RC%
)

if exist Output\PLEAP_Setup_v1.12.exe (
    echo ============================================================
    echo  DONE.  Installer: Output\PLEAP_Setup_v1.12.exe
    echo ============================================================
) else (
    echo [WARN] ISCC reported success but Output\PLEAP_Setup_v1.12.exe not found.
    echo Check the ISCC output above for the actual file location.
)
pause
endlocal
