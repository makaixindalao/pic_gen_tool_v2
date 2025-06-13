@echo off
chcp 65001 >nul
title 军事目标数据集生成平台 - 启动脚本

echo ========================================
echo    军事目标数据集生成与管理平台
echo ========================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [信息] 检测到Python环境
echo.

:: 检查依赖是否安装
echo [信息] 检查后端依赖...
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [警告] 后端依赖未完全安装，正在安装...
    echo [提示] 首次运行需要安装依赖，请稍等...
    cd backend
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 后端依赖安装失败，请检查网络连接
        pause
        exit /b 1
    )
    cd ..
    echo [信息] 后端依赖安装完成
) else (
    echo [信息] 后端依赖检查通过
)

echo [信息] 检查前端依赖...
python -c "import PyQt6, requests" >nul 2>&1
if errorlevel 1 (
    echo [警告] 前端依赖未完全安装，正在安装...
    echo [提示] 首次运行需要安装依赖，请稍等...
    cd frontend
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 前端依赖安装失败，请检查网络连接
        pause
        exit /b 1
    )
    cd ..
    echo [信息] 前端依赖安装完成
) else (
    echo [信息] 前端依赖检查通过
)

echo.
echo [信息] 所有依赖检查完成
echo.

:: 启动后端服务
echo [启动] 正在启动后端服务...
start "后端服务" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

:: 等待后端服务启动
echo [等待] 等待后端服务启动...
timeout /t 8 /nobreak >nul

:: 检查后端服务是否启动成功
:check_backend
python -c "import requests; requests.get('http://127.0.0.1:8000/health', timeout=1)" >nul 2>&1
if errorlevel 1 (
    echo [等待] 后端服务启动中...
    timeout /t 2 /nobreak >nul
    goto check_backend
)

echo [成功] 后端服务启动成功！
echo.

:: 启动前端应用
echo [启动] 正在启动前端应用...
timeout /t 1 /nobreak >nul
start "前端应用" cmd /k "cd /d %~dp0frontend && python src/main_window.py"

echo.
echo ========================================
echo [完成] 服务启动完成！
echo.
echo 后端服务地址: http://127.0.0.1:8000
echo 前端应用已启动
echo.
echo 按任意键关闭此窗口...
echo ========================================
pause >nul 