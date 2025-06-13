@echo off
chcp 65001 >nul
title 军事目标数据集生成平台 - 快速启动

echo ========================================
echo    军事目标数据集生成平台 - 快速启动
echo ========================================
echo.

echo [启动] 正在启动后端服务...
start "后端服务" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [等待] 等待后端服务启动...
timeout /t 8 /nobreak >nul

echo [启动] 正在启动前端应用...
start "前端应用" cmd /k "cd /d %~dp0frontend && python src/main_window.py"

echo.
echo ========================================
echo [完成] 服务启动完成！
echo.
echo 后端服务: http://127.0.0.1:8000
echo 前端应用已启动
echo.
echo 如需停止服务，请运行 stop.bat
echo ========================================
echo.
timeout /t 3 /nobreak >nul 