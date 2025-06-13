@echo off
chcp 65001 >nul
title 停止服务 - 军事目标数据集生成平台

echo ========================================
echo    停止军事目标数据集生成平台服务
echo ========================================
echo.

echo [停止] 正在停止后端服务...
taskkill /f /im python.exe /fi "WINDOWTITLE eq 后端服务*" >nul 2>&1
taskkill /f /im uvicorn.exe >nul 2>&1

echo [停止] 正在停止前端应用...
taskkill /f /im python.exe /fi "WINDOWTITLE eq 前端应用*" >nul 2>&1

:: 更彻底的清理
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo table /nh ^| findstr uvicorn') do (
    taskkill /f /pid %%i >nul 2>&1
)

for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo table /nh ^| findstr main_window') do (
    taskkill /f /pid %%i >nul 2>&1
)

echo.
echo [完成] 所有服务已停止
echo.
pause 