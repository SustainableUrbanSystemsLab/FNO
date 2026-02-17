@echo off
REM Single GPU Training (Original Script)
REM Uses train_fno_mag.py with sequential data loading

echo ==========================================
echo  FNO Training - Single GPU (Original)
echo ==========================================

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Starting training...
python train_fno_mag.py

echo.
echo Training complete!
pause
