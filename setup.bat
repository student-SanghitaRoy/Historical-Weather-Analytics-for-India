@echo off
cd /d "%~dp0"
echo === Step 1: creating virtual environment ===
python -m venv venv
if errorlevel 1 (echo Python was not found. Install Python 3.10+ and tick "Add to PATH". & pause & exit /b 1)
call venv\Scripts\activate.bat
echo === Step 2: installing libraries (a few minutes) ===
python -m pip install -r requirements.txt
echo === Step 3: running tests (about 1 minute) ===
python -m pytest -q
echo.
echo If the last line above says "passed", setup worked. Next: double-click download_data.bat, then run_app.bat
pause
