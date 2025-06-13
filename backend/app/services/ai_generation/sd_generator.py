"""
Stable Diffusion 图像生成器
实现基于扩散模型的图像生成功能
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMScheduler
)
from PIL import Image

logger = logging.getLogger(__name__)

class SDGenerator:
    """Stable Diffusion 图像生成器"""
    
    # 支持的采样器映射
    SCHEDULERS = {
        "DPM++ 2M Karras": DPMSolverMultistepScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
        "Euler": EulerDiscreteScheduler,
        "LMS": LMSDiscreteScheduler,
        "Heun": HeunDiscreteScheduler,
        "DPM2": KDPM2DiscreteScheduler,
        "DPM2 a": KDPM2AncestralDiscreteScheduler,
        "DPM++ 2S a": DPMSolverSinglestepScheduler,
        "DDIM": DDIMScheduler,
    }
    
    def __init__(self, model_path: str = "runwayml/stable-diffusion-v1-5", device: str = "auto"):
        """
        初始化SD生成器
        
        Args:
            model_path: 模型路径或HuggingFace模型ID
            device: 设备类型，"auto"自动选择
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.pipe = None
        self.current_scheduler = None
        
        logger.info(f"初始化SD生成器，模型: {model_path}, 设备: {self.device}")
        
    def _get_device(self, device: str) -> str:
        """获取可用设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self) -> bool:
        """
        加载模型
        
        Returns:
            bool: 加载是否成功
        """
        try:
            import torch
            logger.info("开始加载Stable Diffusion模型...")
            
            # 根据设备选择数据类型
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # 检查是否为本地路径
            local_model_path = Path(self.model_path)
            if local_model_path.exists() and local_model_path.is_dir():
                logger.info(f"从本地路径加载模型: {self.model_path}")
                model_source = str(local_model_path.resolve())  # 使用绝对路径
                use_local_files_only = True
            else:
                # 检查相对路径
                relative_path = Path("models/stable-diffusion-v1-5")
                if relative_path.exists() and relative_path.is_dir():
                    logger.info(f"从本地相对路径加载模型: {relative_path}")
                    model_source = str(relative_path.resolve())
                    use_local_files_only = True
                else:
                    logger.info(f"从Hugging Face Hub加载模型: {self.model_path}")
                    model_source = self.model_path
                    use_local_files_only = False
            
            # 尝试加载模型
            try:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_source,
                    torch_dtype=torch_dtype,
                    safety_checker=None,  # 禁用安全检查器以提高速度
                    requires_safety_checker=False,
                    local_files_only=use_local_files_only  # 如果是本地路径，只使用本地文件
                )
            except Exception as e:
                if "Connection error" in str(e) or "Max retries exceeded" in str(e):
                    logger.warning("网络连接失败，尝试使用本地缓存...")
                    # 尝试从本地缓存加载
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=True  # 强制使用本地文件
                    )
                else:
                    raise e
            
            # 移动到指定设备
            self.pipe = self.pipe.to(self.device)
            
            # 启用内存优化
            if self.device == "cuda":
                # 对于高端GPU（16GB显存），禁用CPU卸载以获得最佳性能
                try:
                    import torch
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if total_memory >= 12:  # 12GB以上显存
                        logger.info(f"检测到高端GPU ({total_memory:.1f}GB)，禁用CPU卸载以获得最佳性能")
                        # 不启用CPU卸载，保持模型完全在GPU上
                        self.pipe.enable_attention_slicing(1)  # 轻微的注意力切片
                    else:
                        logger.info(f"中低端GPU ({total_memory:.1f}GB)，启用CPU卸载节省显存")
                        self.pipe.enable_attention_slicing()
                        self.pipe.enable_model_cpu_offload()
                except Exception as e:
                    logger.warning(f"无法检测GPU显存，使用默认优化: {str(e)}")
                    self.pipe.enable_attention_slicing()
                    self.pipe.enable_model_cpu_offload()
            
            logger.info("模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            # 提供详细的错误信息和解决建议
            if "Connection error" in str(e) or "Max retries exceeded" in str(e):
                logger.error("网络连接问题，请检查网络连接或使用本地模型")
                logger.error("解决方案：")
                logger.error("1. 检查网络连接")
                logger.error("2. 使用模型下载脚本: python scripts/download_models.py")
                logger.error("3. 配置本地模型路径")
            elif "not cached locally" in str(e):
                logger.error("模型未在本地缓存，且无法从网络下载")
                logger.error("解决方案：")
                logger.error("1. 运行模型下载脚本: python scripts/download_models.py")
                logger.error("2. 手动下载模型到 models/stable-diffusion-v1-5 目录")
            return False
    
    def set_scheduler(self, scheduler_name: str):
        """
        设置采样器
        
        Args:
            scheduler_name: 采样器名称
        """
        if not self.pipe:
            raise RuntimeError("模型未加载，请先调用load_model()")
            
        if scheduler_name not in self.SCHEDULERS:
            raise ValueError(f"不支持的采样器: {scheduler_name}")
        
        scheduler_class = self.SCHEDULERS[scheduler_name]
        
        # 获取当前调度器的配置
        scheduler_config = self.pipe.scheduler.config
        
        # 创建新的调度器
        if scheduler_name in ["DPM++ 2M Karras", "LMS Karras", "DPM2 Karras", "DPM2 a Karras", "DPM++ 2S a Karras"]:
            # Karras调度器需要特殊配置
            self.pipe.scheduler = scheduler_class.from_config(
                scheduler_config, 
                use_karras_sigmas=True
            )
        else:
            self.pipe.scheduler = scheduler_class.from_config(scheduler_config)
        
        self.current_scheduler = scheduler_name
        logger.info(f"采样器已设置为: {scheduler_name}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        steps: int = 30,
        cfg_scale: float = 7.5,
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        scheduler_name: str = "DPM++ 2M Karras"
    ) -> List[Image.Image]:
        """
        生成图像
        
        Args:
            prompt: 正面提示词
            negative_prompt: 负面提示词
            num_images: 生成图像数量
            steps: 采样步数
            cfg_scale: CFG引导强度
            seed: 随机种子，-1表示随机
            width: 图像宽度
            height: 图像高度
            scheduler_name: 采样器名称
            
        Returns:
            List[Image.Image]: 生成的图像列表
        """
        if not self.pipe:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        # 设置采样器
        if scheduler_name != self.current_scheduler:
            self.set_scheduler(scheduler_name)
        
        # 处理种子
        if seed == -1:
            import torch
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        import torch
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"开始生成图像 - 提示词: {prompt[:50]}..., 步数: {steps}, CFG: {cfg_scale}, 种子: {seed}")
        
        try:
            # 生成图像
            import torch
            with torch.autocast(self.device if self.device != "mps" else "cpu"):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    width=width,
                    height=height
                )
            
            images = result.images
            logger.info(f"成功生成 {len(images)} 张图像，使用设备: {self.device}")
            return images
            
        except Exception as e:
            logger.error(f"图像生成失败: {str(e)}")
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[List[Image.Image]]:
        """
        批量生成图像
        
        Args:
            prompts: 提示词列表
            **kwargs: 其他生成参数
            
        Returns:
            List[List[Image.Image]]: 每个提示词对应的图像列表
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"批量生成进度: {i+1}/{len(prompts)}")
            images = self.generate(prompt=prompt, **kwargs)
            results.append(images)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "current_scheduler": self.current_scheduler,
            "available_schedulers": list(self.SCHEDULERS.keys()),
            "is_loaded": self.pipe is not None
        }
    
    def unload_model(self):
        """卸载模型以释放内存"""
        if self.pipe:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("模型已卸载")
