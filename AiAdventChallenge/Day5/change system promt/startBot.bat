@echo off
chcp 65001 >nul

echo ======================================
echo Запуск web AI интерфейса
echo ======================================
 
pip install -r requirements.txt

echo ▶ Запуск приложения...
uvicorn web_app:app --reload
pause
