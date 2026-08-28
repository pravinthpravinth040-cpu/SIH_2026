@echo off
title OceanGuard Maritime Backend Server
echo ========================================================
echo   OCEANGUARD AI Maritime Oil Spill Monitoring Backend
echo ========================================================
echo.
cd /d "%~dp0"
echo Starting FastAPI Backend on http://localhost:8000 ...
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
