@echo off
REM Multi-GPU Training (2 GPUs)
REM Uses torchrun with train_fno_distributed.py

echo ==========================================
echo  FNO Training - Multi-GPU (2 GPUs)
echo ==========================================

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Starting distributed training on 2 GPUs...
torchrun --nproc_per_node=2 train_fno_distributed.py

echo.
echo Training complete!
pause
