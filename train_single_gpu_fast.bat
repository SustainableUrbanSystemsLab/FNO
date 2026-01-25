@echo off
REM Single GPU Training (Distributed Script - with parallel data loading)
REM Uses train_fno_distributed.py but falls back to single GPU mode

echo ==========================================
echo  FNO Training - Single GPU (+ Fast Data Loading)
echo ==========================================

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Starting training with parallel data loading...
python train_fno_distributed.py

echo.
echo Training complete!
pause
