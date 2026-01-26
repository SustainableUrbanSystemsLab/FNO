@echo off
REM Run Inference on test_csv folder
REM Uses run_inference_mag.py

echo ==========================================
echo  FNO Inference
echo ==========================================

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Running inference on test_csv folder...
python run_inference_mag.py

echo.
echo Inference complete!
pause
