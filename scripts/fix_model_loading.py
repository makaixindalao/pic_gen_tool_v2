#!/usr/bin/env python3
"""
模型加载问题修复脚本
快速诊断和修复Stable Diffusion模型加载问题
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_network_connection():
    """检查网络连接"""
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_local_model():
    """检查本地模型是否存在"""
    model_paths = [
        Path("models/stable-diffusion-v1-5"),
        Path.home() / ".cache" / "huggingface" / "transformers" / "runwayml--stable-diffusion-v1-5"
    ]
    
    for path in model_paths:
        if path.exists() and path.is_dir():
            # 检查关键文件
            required_files = [
                "model_index.json",
                "unet/config.json",
                "text_encoder/config.json",
                "vae/config.json"
            ]
            
            if all((path / file).exists() for file in required_files):
                logger.info(f"找到完整的本地模型: {path}")
                return str(path)
    
    return None

def install_dependencies():
    """安装必要的依赖"""
    try:
        logger.info("检查并安装必要的依赖...")
        
        # 检查PyTorch
        try:
            import torch
            logger.info(f"PyTorch版本: {torch.__version__}")
            logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        except ImportError:
            logger.error("PyTorch未安装，请先安装PyTorch")
            return False
        
        # 检查diffusers
        try:
            import diffusers
            logger.info(f"Diffusers版本: {diffusers.__version__}")
        except ImportError:
            logger.info("安装diffusers...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "diffusers"])
        
        # 检查transformers
        try:
            import transformers
            logger.info(f"Transformers版本: {transformers.__version__}")
        except ImportError:
            logger.info("安装transformers...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        
        return True
        
    except Exception as e:
        logger.error(f"依赖安装失败: {str(e)}")
        return False

def download_model_offline():
    """离线下载模型的方法"""
    logger.info("=== 离线下载模型方法 ===")
    print("""
1. 使用Git LFS下载模型：
   git lfs install
   git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/stable-diffusion-v1-5

2. 使用huggingface-hub下载：
   pip install huggingface-hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download('runwayml/stable-diffusion-v1-5', local_dir='models/stable-diffusion-v1-5')"

3. 手动下载（如果网络不稳定）：
   - 访问 https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main
   - 下载所有文件到 models/stable-diffusion-v1-5 目录
   - 保持原有的目录结构

4. 使用镜像站点：
   - 设置环境变量: export HF_ENDPOINT=https://hf-mirror.com
   - 然后运行正常的下载命令
    """)

def create_local_config():
    """创建本地配置文件"""
    config_content = """# AI模型配置
AI_MODEL_PATH=models/stable-diffusion-v1-5
AI_USE_LOCAL_MODEL=true
AI_DEVICE=auto
AI_TORCH_DTYPE=auto
"""
    
    config_file = Path(".env")
    if not config_file.exists():
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        logger.info(f"创建配置文件: {config_file}")
    else:
        logger.info("配置文件已存在")

def test_model_loading():
    """测试模型加载"""
    try:
        logger.info("测试模型加载...")
        
        # 添加项目路径
        sys.path.append(str(Path(__file__).parent.parent))
        
        from backend.app.services.ai_generation.sd_generator import SDGenerator
        
        # 尝试不同的模型路径
        model_paths = [
            "models/stable-diffusion-v1-5",
            "runwayml/stable-diffusion-v1-5"
        ]
        
        for model_path in model_paths:
            try:
                logger.info(f"尝试加载模型: {model_path}")
                generator = SDGenerator(model_path)
                success = generator.load_model()
                
                if success:
                    logger.info(f"✓ 模型加载成功: {model_path}")
                    generator.unload_model()
                    return True
                else:
                    logger.warning(f"✗ 模型加载失败: {model_path}")
                    
            except Exception as e:
                logger.error(f"✗ 模型加载异常 {model_path}: {str(e)}")
        
        return False
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=== Stable Diffusion 模型加载问题修复工具 ===\n")
    
    # 1. 检查依赖
    logger.info("1. 检查依赖...")
    if not install_dependencies():
        logger.error("依赖检查失败，请手动安装必要的依赖")
        return
    
    # 2. 检查网络连接
    logger.info("2. 检查网络连接...")
    network_ok = check_network_connection()
    if network_ok:
        logger.info("✓ 网络连接正常")
    else:
        logger.warning("✗ 网络连接异常，将使用离线模式")
    
    # 3. 检查本地模型
    logger.info("3. 检查本地模型...")
    local_model = check_local_model()
    if local_model:
        logger.info(f"✓ 找到本地模型: {local_model}")
    else:
        logger.warning("✗ 未找到本地模型")
        
        if not network_ok:
            download_model_offline()
            return
    
    # 4. 创建配置文件
    logger.info("4. 创建配置文件...")
    create_local_config()
    
    # 5. 测试模型加载
    logger.info("5. 测试模型加载...")
    if test_model_loading():
        logger.info("✓ 模型加载测试通过！")
        print("\n=== 修复完成 ===")
        print("现在可以正常使用AI生成功能了")
    else:
        logger.error("✗ 模型加载测试失败")
        print("\n=== 需要手动处理 ===")
        print("请按照以下步骤手动下载模型：")
        download_model_offline()

if __name__ == "__main__":
    main() 