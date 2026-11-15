@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo This downloads weather data for all 71 locations and waits out the hourly limit by itself.
echo Keep this window open and the computer awake. Safe to close and run again later.
python scripts\prefetch_india.py --auto
pause
