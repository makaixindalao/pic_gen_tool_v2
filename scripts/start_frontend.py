#!/usr/bin/env python3
"""
前端应用启动脚本
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="启动前端GUI应用")
    parser.add_argument("--backend-url", default="http://localhost:8000", 
                       help="后端服务地址")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    frontend_dir = project_root / "frontend"
    
    # 切换到前端目录
    os.chdir(frontend_dir)
    
    # 检查是否安装了依赖
    requirements_file = frontend_dir / "requirements.txt"
    if requirements_file.exists():
        print("检查依赖安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True)
            print("依赖检查完成")
        except subprocess.CalledProcessError:
            print("警告: 依赖安装可能有问题")
    
    # 设置环境变量
    env = os.environ.copy()
    env["BACKEND_URL"] = args.backend_url
    
    if args.debug:
        env["DEBUG"] = "1"
    
    # 启动前端应用
    main_script = frontend_dir / "src" / "main_window.py"
    
    if not main_script.exists():
        print(f"错误: 找不到主程序文件 {main_script}")
        return 1
    
    print(f"启动前端应用...")
    print(f"后端服务地址: {args.backend_url}")
    print("关闭窗口或按 Ctrl+C 退出应用")
    
    try:
        subprocess.run([sys.executable, str(main_script)], env=env)
    except KeyboardInterrupt:
        print("\n应用已退出")
    except Exception as e:
        print(f"启动失败: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 