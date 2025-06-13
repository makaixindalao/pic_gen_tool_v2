#!/usr/bin/env python3
"""
军事目标数据集生成平台启动脚本
"""

import sys
import os
import subprocess
import argparse
import time
import signal
from pathlib import Path
from threading import Thread

def start_backend(host="localhost", port=8000, reload=False):
    """启动后端服务"""
    backend_script = Path(__file__).parent / "scripts" / "start_backend.py"
    
    cmd = [sys.executable, str(backend_script), "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    
    print(f"启动后端服务: {host}:{port}")
    return subprocess.Popen(cmd)

def start_frontend(backend_url="http://localhost:8000", debug=False):
    """启动前端应用"""
    frontend_script = Path(__file__).parent / "scripts" / "start_frontend.py"
    
    cmd = [sys.executable, str(frontend_script), "--backend-url", backend_url]
    if debug:
        cmd.append("--debug")
    
    print(f"启动前端应用，连接到: {backend_url}")
    return subprocess.Popen(cmd)

def wait_for_backend(host, port, timeout=30):
    """等待后端服务启动"""
    import requests
    
    url = f"http://{host}:{port}/health"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print("后端服务已就绪")
                return True
        except:
            pass
        time.sleep(1)
    
    print("等待后端服务超时")
    return False

def main():
    parser = argparse.ArgumentParser(description="军事目标数据集生成平台")
    parser.add_argument("--mode", choices=["all", "backend", "frontend"], 
                       default="all", help="启动模式")
    parser.add_argument("--host", default="localhost", help="后端服务主机")
    parser.add_argument("--port", type=int, default=8000, help="后端服务端口")
    parser.add_argument("--reload", action="store_true", help="后端自动重载")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    processes = []
    
    try:
        if args.mode in ["all", "backend"]:
            # 启动后端服务
            backend_process = start_backend(args.host, args.port, args.reload)
            processes.append(backend_process)
            
            if args.mode == "all":
                # 等待后端服务启动
                print("等待后端服务启动...")
                if not wait_for_backend(args.host, args.port):
                    print("后端服务启动失败")
                    return 1
        
        if args.mode in ["all", "frontend"]:
            # 启动前端应用
            backend_url = f"http://{args.host}:{args.port}"
            frontend_process = start_frontend(backend_url, args.debug)
            processes.append(frontend_process)
        
        if args.mode == "backend":
            print(f"后端服务运行在: http://{args.host}:{args.port}")
            print("API文档: http://{args.host}:{args.port}/docs")
            print("按 Ctrl+C 停止服务")
        
        # 等待进程结束
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        
        # 终止所有进程
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print("所有服务已停止")
    
    except Exception as e:
        print(f"启动失败: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 