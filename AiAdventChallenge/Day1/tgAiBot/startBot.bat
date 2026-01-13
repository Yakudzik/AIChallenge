@echo off
chcp 65001 >nul

echo ======================================
echo Запуск Telegram AI бота
echo ======================================
 
pip install -r requirements.txt

echo ▶ Запуск бота...
py bot.py

pause