#!/usr/bin/env python3
"""
后端服务启动脚本
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="启动后端AI生成服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用自动重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数量")
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"
    
    # 切换到后端目录
    os.chdir(backend_dir)
    
    # 检查是否安装了依赖
    requirements_file = backend_dir / "requirements.txt"
    if requirements_file.exists():
        print("检查依赖安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True)
            print("依赖检查完成")
        except subprocess.CalledProcessError:
            print("警告: 依赖安装可能有问题")
    
    # 构建启动命令
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    if args.workers > 1:
        cmd.extend(["--workers", str(args.workers)])
    
    print(f"启动后端服务: {' '.join(cmd)}")
    print(f"服务地址: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n服务已停止")

if __name__ == "__main__":
    main() 