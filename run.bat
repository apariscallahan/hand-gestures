@echo off
rem Starts Gesture Control, installing what it needs the first time.
rem Arguments are passed on, e.g.  run.bat --dry-run   or   run.bat --list-cameras
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up a private Python environment. This takes a minute or two...
    python -m venv .venv || goto :no_python
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :install_failed
)

".venv\Scripts\python.exe" -m gesture_control %*
if errorlevel 1 pause
exit /b

:no_python
echo.
echo Python 3.11 or newer is needed. Get it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" in the installer), then run this again.
pause
exit /b 1

:install_failed
echo.
echo Installing the required packages failed - see the messages above.
rmdir /s /q .venv
pause
exit /b 1
