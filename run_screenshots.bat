@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python run_step_by_step.py
pause
