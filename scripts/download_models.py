#!/usr/bin/env python3
"""
模型下载脚本
用于手动下载Stable Diffusion模型，解决网络连接问题
"""

import os
import sys
import requests
import json
from pathlib import Path
from typing import Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDownloader:
    """模型下载器"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # 模型配置
        self.model_configs = {
            "stable-diffusion-v1-5": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "local_path": self.models_dir / "stable-diffusion-v1-5",
                "files": [
                    "model_index.json",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder/pytorch_model.bin",
                    "tokenizer/merges.txt",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.bin",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.bin",
                    "feature_extractor/preprocessor_config.json",
                    "safety_checker/config.json",
                    "safety_checker/pytorch_model.bin"
                ],
                "base_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/"
            }
        }
    
    def download_file(self, url: str, local_path: Path, timeout: int = 300) -> bool:
        """下载单个文件"""
        try:
            logger.info(f"下载文件: {url}")
            
            # 创建目录
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载文件
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"下载完成: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"下载失败 {url}: {str(e)}")
            return False
    
    def download_model(self, model_name: str) -> bool:
        """下载指定模型"""
        if model_name not in self.model_configs:
            logger.error(f"未知模型: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        logger.info(f"开始下载模型: {config['model_id']}")
        
        success_count = 0
        total_files = len(config['files'])
        
        for file_path in config['files']:
            url = config['base_url'] + file_path
            local_path = config['local_path'] / file_path
            
            # 检查文件是否已存在
            if local_path.exists():
                logger.info(f"文件已存在，跳过: {local_path}")
                success_count += 1
                continue
            
            if self.download_file(url, local_path):
                success_count += 1
            else:
                logger.warning(f"文件下载失败: {file_path}")
        
        logger.info(f"下载完成: {success_count}/{total_files} 个文件")
        return success_count == total_files
    
    def verify_model(self, model_name: str) -> bool:
        """验证模型完整性"""
        if model_name not in self.model_configs:
            return False
        
        config = self.model_configs[model_name]
        missing_files = []
        
        for file_path in config['files']:
            local_path = config['local_path'] / file_path
            if not local_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"缺失文件: {missing_files}")
            return False
        
        logger.info(f"模型验证通过: {model_name}")
        return True
    
    def create_symlink(self, model_name: str) -> bool:
        """创建符号链接到Hugging Face缓存目录"""
        try:
            import transformers
            from huggingface_hub import snapshot_download
            
            config = self.model_configs[model_name]
            source_path = config['local_path']
            
            # 获取Hugging Face缓存目录
            cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建模型缓存目录
            model_cache_dir = cache_dir / config['model_id'].replace('/', '--')
            
            if model_cache_dir.exists():
                logger.info(f"缓存目录已存在: {model_cache_dir}")
                return True
            
            # 创建符号链接
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copytree(source_path, model_cache_dir)
                logger.info(f"复制模型到缓存目录: {model_cache_dir}")
            else:  # Linux/Mac
                model_cache_dir.symlink_to(source_path)
                logger.info(f"创建符号链接: {model_cache_dir} -> {source_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"创建缓存链接失败: {str(e)}")
            return False

def main():
    """主函数"""
    downloader = ModelDownloader()
    
    print("=== Stable Diffusion 模型下载工具 ===")
    print("1. 下载 Stable Diffusion v1.5")
    print("2. 验证模型完整性")
    print("3. 创建缓存链接")
    print("4. 全部执行")
    print("0. 退出")
    
    while True:
        choice = input("\n请选择操作 (0-4): ").strip()
        
        if choice == "0":
            print("退出程序")
            break
        elif choice == "1":
            print("\n开始下载模型...")
            success = downloader.download_model("stable-diffusion-v1-5")
            if success:
                print("✓ 模型下载完成")
            else:
                print("✗ 模型下载失败")
        elif choice == "2":
            print("\n验证模型完整性...")
            if downloader.verify_model("stable-diffusion-v1-5"):
                print("✓ 模型验证通过")
            else:
                print("✗ 模型验证失败")
        elif choice == "3":
            print("\n创建缓存链接...")
            if downloader.create_symlink("stable-diffusion-v1-5"):
                print("✓ 缓存链接创建成功")
            else:
                print("✗ 缓存链接创建失败")
        elif choice == "4":
            print("\n执行完整流程...")
            
            # 下载模型
            print("1/3 下载模型...")
            if not downloader.download_model("stable-diffusion-v1-5"):
                print("✗ 模型下载失败")
                continue
            
            # 验证模型
            print("2/3 验证模型...")
            if not downloader.verify_model("stable-diffusion-v1-5"):
                print("✗ 模型验证失败")
                continue
            
            # 创建缓存链接
            print("3/3 创建缓存链接...")
            if downloader.create_symlink("stable-diffusion-v1-5"):
                print("✓ 所有步骤完成！")
            else:
                print("✗ 缓存链接创建失败")
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 