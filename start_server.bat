@echo off
echo Starting OceanGuard AI Backend Server...
echo To create a new database, run: create_database.bat
cd /d "%~dp0backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
