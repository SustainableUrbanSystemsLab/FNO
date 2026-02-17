@echo off
REM Generate publication-ready plots from training logs
REM Usage: generate_plots.bat [optional_path_to_metrics.csv]

echo ==========================================
echo  FNO Publication Plots Generator
echo ==========================================

cd /d "%~dp0"

if "%~1"=="" (
    echo Looking for latest training run...
    python generate_plots.py
) else (
    echo Processing: %~1
    python generate_plots.py "%~1"
)

echo.
echo Done! Check the training_logs folder for PNG files.
pause
