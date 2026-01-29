@echo off
chcp 65001 >nul

echo ======================================
echo Запуск web AI интерфейса
echo ======================================
 
python -m pip install -r requirements.txt

echo ▶ Запуск приложения...
cd /d "change system promt"

set PYTHONLOGLEVEL=DEBUG
python -m uvicorn web_app:app --reload 
@--log-level debug
pause
