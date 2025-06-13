from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
from functools import lru_cache

class Settings(BaseSettings):
    """应用配置"""
    
    # 基础配置
    app_name: str = Field(default="军事目标数据集生成平台", description="应用名称")
    debug: bool = Field(default=True, description="调试模式")
    host: str = Field(default="0.0.0.0", description="服务器主机")
    port: int = Field(default=8000, description="服务器端口")
    
    # 数据库配置
    database_url: str = Field(default="sqlite:///./data/database.db", description="数据库URL")
    
    # 文件存储配置
    upload_dir: str = Field(default="data/uploads", description="上传目录")
    output_dir: str = Field(default="data/generated", description="输出目录")
    max_file_size: int = Field(default=10485760, description="最大文件大小(字节)")
    
    # AI生成配置
    ai_model_path: str = Field(
        default="runwayml/stable-diffusion-v1-5",
        description="Stable Diffusion模型路径或HuggingFace模型ID"
    )
    ai_local_models_dir: str = Field(
        default="models",
        description="本地模型存储目录"
    )
    ai_use_local_model: bool = Field(
        default=True,
        description="优先使用本地模型"
    )
    ai_device: str = Field(
        default="auto",
        description="AI推理设备 (auto/cuda/cpu/mps)"
    )
    ai_enable_model_offload: bool = Field(
        default=True,
        description="启用模型CPU卸载以节省显存"
    )
    ai_torch_dtype: str = Field(
        default="auto",
        description="PyTorch数据类型 (auto/float16/float32)"
    )
    
    # YOLO检测配置
    yolo_model_path: str = Field(default="yolov8n.pt", description="YOLO模型路径")
    yolo_confidence_threshold: float = Field(default=0.5, description="YOLO检测置信度阈值")
    yolo_nms_threshold: float = Field(default=0.4, description="YOLO NMS阈值")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: str = Field(default="logs/app.log", description="日志文件路径")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_ai_model_path(self) -> str:
        """获取AI模型路径"""
        if self.ai_use_local_model:
            local_path = Path(self.ai_local_models_dir) / "stable-diffusion-v1-5"
            if local_path.exists() and local_path.is_dir():
                return str(local_path)
        return self.ai_model_path
    
    def get_ai_device(self) -> str:
        """获取AI推理设备"""
        if self.ai_device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.ai_device
    
    def get_torch_dtype(self):
        """获取PyTorch数据类型"""
        if self.ai_torch_dtype == "auto":
            import torch
            device = self.get_ai_device()
            return torch.float16 if device == "cuda" else torch.float32
        elif self.ai_torch_dtype == "float16":
            import torch
            return torch.float16
        else:
            import torch
            return torch.float32

@lru_cache()
def get_settings() -> Settings:
    """获取应用设置（单例）"""
    return Settings()
