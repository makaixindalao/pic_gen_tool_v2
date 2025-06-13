"""
文件管理器
用于管理数据集文件的组织和存储
"""

import os
import shutil
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FileManager:
    """文件管理器"""
    
    def __init__(self, base_dir: str = "data"):
        """
        初始化文件管理器
        
        Args:
            base_dir: 基础数据目录
        """
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.generated_dir = self.base_dir / "generated"
        self.raw_assets_dir = self.base_dir / "raw_assets"
        self.temp_dir = self.base_dir / "temp"
        
        # 创建必要的目录结构
        self._create_directory_structure()
        
        logger.info(f"文件管理器初始化完成，基础目录: {self.base_dir}")
    
    def _create_directory_structure(self):
        """创建目录结构"""
        directories = [
            self.datasets_dir,
            self.generated_dir / "ai",
            self.generated_dir / "traditional",
            self.raw_assets_dir / "targets",
            self.raw_assets_dir / "backgrounds",
            self.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"确保目录存在: {directory}")
    
    def create_dataset_directory(
        self,
        dataset_name: str,
        generation_type: str = "ai"
    ) -> Path:
        """
        创建数据集目录
        
        Args:
            dataset_name: 数据集名称
            generation_type: 生成类型 (ai/traditional)
            
        Returns:
            Path: 数据集目录路径
        """
        # 生成唯一的数据集目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self._sanitize_filename(dataset_name)
        dataset_dir_name = f"{safe_name}_{generation_type}_{timestamp}"
        
        dataset_path = self.datasets_dir / dataset_dir_name
        
        # 创建数据集子目录
        subdirs = ["images", "annotations", "metadata"]
        for subdir in subdirs:
            (dataset_path / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建数据集目录: {dataset_path}")
        return dataset_path
    
    def save_generated_image(
        self,
        image_data: bytes,
        filename: str,
        generation_type: str = "ai",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        保存生成的图像
        
        Args:
            image_data: 图像数据
            filename: 文件名
            generation_type: 生成类型
            metadata: 元数据
            
        Returns:
            Tuple[Path, Dict]: (文件路径, 文件信息)
        """
        # 确保文件名安全
        safe_filename = self._sanitize_filename(filename)
        
        # 生成文件路径
        output_dir = self.generated_dir / generation_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / safe_filename
        
        # 保存图像文件
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        # 生成文件信息
        file_info = {
            "filename": safe_filename,
            "path": str(file_path),
            "size": len(image_data),
            "hash": self._calculate_hash(image_data),
            "created_at": datetime.now().isoformat(),
            "generation_type": generation_type,
            "metadata": metadata or {}
        }
        
        # 保存元数据文件
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(file_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存生成图像: {file_path}")
        return file_path, file_info
    
    def save_annotation(
        self,
        annotation: Dict[str, Any],
        image_path: str,
        annotation_format: str = "coco"
    ) -> Path:
        """
        保存标注文件
        
        Args:
            annotation: 标注数据
            image_path: 对应的图像路径
            annotation_format: 标注格式
            
        Returns:
            Path: 标注文件路径
        """
        # 生成标注文件名
        image_path_obj = Path(image_path)
        annotation_filename = image_path_obj.stem + "_annotation.json"
        
        # 确定保存目录
        if "ai_generated" in str(image_path):
            annotation_dir = self.generated_dir / "ai_generated" / "annotations"
        elif "traditional" in str(image_path):
            annotation_dir = self.generated_dir / "traditional" / "annotations"
        else:
            annotation_dir = self.generated_dir / "annotations"
        
        annotation_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = annotation_dir / annotation_filename
        
        # 保存标注文件
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存标注文件: {annotation_path}")
        return annotation_path
    
    def organize_dataset(
        self,
        source_files: List[Path],
        dataset_path: Path,
        copy_files: bool = True
    ) -> Dict[str, Any]:
        """
        组织数据集文件
        
        Args:
            source_files: 源文件列表
            dataset_path: 数据集目录
            copy_files: 是否复制文件（否则移动）
            
        Returns:
            Dict: 组织结果
        """
        images_dir = dataset_path / "images"
        annotations_dir = dataset_path / "annotations"
        
        organized_files = {
            "images": [],
            "annotations": [],
            "metadata": []
        }
        
        for source_file in source_files:
            if not source_file.exists():
                logger.warning(f"源文件不存在: {source_file}")
                continue
            
            # 根据文件扩展名分类
            if source_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                target_dir = images_dir
                category = "images"
            elif source_file.suffix.lower() == '.json':
                target_dir = annotations_dir
                category = "annotations"
            else:
                target_dir = dataset_path / "metadata"
                category = "metadata"
            
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_file.name
            
            # 复制或移动文件
            if copy_files:
                shutil.copy2(source_file, target_path)
            else:
                shutil.move(str(source_file), str(target_path))
            
            organized_files[category].append(str(target_path))
        
        # 生成数据集清单
        manifest = {
            "dataset_name": dataset_path.name,
            "created_at": datetime.now().isoformat(),
            "total_files": sum(len(files) for files in organized_files.values()),
            "files": organized_files
        }
        
        manifest_path = dataset_path / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集组织完成: {dataset_path}")
        return manifest
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        清理临时文件
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        if not self.temp_dir.exists():
            return
        
        current_time = datetime.now()
        cleaned_count = 0
        
        for file_path in self.temp_dir.rglob('*'):
            if file_path.is_file():
                # 检查文件年龄
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {file_path}, {str(e)}")
        
        logger.info(f"清理临时文件完成，删除 {cleaned_count} 个文件")
    
    def get_dataset_info(self, dataset_path: Path) -> Optional[Dict[str, Any]]:
        """
        获取数据集信息
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            Dict: 数据集信息
        """
        manifest_path = dataset_path / "manifest.json"
        
        if not manifest_path.exists():
            return None
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取数据集信息失败: {str(e)}")
            return None
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        列出所有数据集
        
        Returns:
            List[Dict]: 数据集列表
        """
        datasets = []
        
        if not self.datasets_dir.exists():
            return datasets
        
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                info = self.get_dataset_info(dataset_dir)
                if info:
                    info["path"] = str(dataset_dir)
                    datasets.append(info)
        
        # 按创建时间排序
        datasets.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return datasets
    
    def delete_dataset(self, dataset_path: Path) -> bool:
        """
        删除数据集
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            bool: 是否删除成功
        """
        try:
            if dataset_path.exists() and dataset_path.is_dir():
                shutil.rmtree(dataset_path)
                logger.info(f"删除数据集: {dataset_path}")
                return True
            else:
                logger.warning(f"数据集不存在: {dataset_path}")
                return False
        except Exception as e:
            logger.error(f"删除数据集失败: {str(e)}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            Dict: 存储统计
        """
        stats = {
            "total_size": 0,
            "datasets_count": 0,
            "images_count": 0,
            "annotations_count": 0
        }
        
        if self.base_dir.exists():
            for file_path in self.base_dir.rglob('*'):
                if file_path.is_file():
                    stats["total_size"] += file_path.stat().st_size
                    
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        stats["images_count"] += 1
                    elif file_path.suffix.lower() == '.json' and 'annotation' in file_path.name:
                        stats["annotations_count"] += 1
        
        if self.datasets_dir.exists():
            stats["datasets_count"] = len([d for d in self.datasets_dir.iterdir() if d.is_dir()])
        
        # 转换大小为可读格式
        stats["total_size_mb"] = round(stats["total_size"] / (1024 * 1024), 2)
        
        return stats
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名，移除不安全字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            str: 安全的文件名
        """
        # 移除或替换不安全字符
        unsafe_chars = '<>:"/\\|?*'
        safe_filename = filename
        
        for char in unsafe_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # 限制长度
        if len(safe_filename) > 200:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:200-len(ext)] + ext
        
        return safe_filename
    
    def _calculate_hash(self, data: bytes) -> str:
        """
        计算数据的哈希值
        
        Args:
            data: 数据
            
        Returns:
            str: MD5哈希值
        """
        return hashlib.md5(data).hexdigest()
    
    def create_temp_file(self, suffix: str = ".tmp") -> Path:
        """
        创建临时文件
        
        Args:
            suffix: 文件后缀
            
        Returns:
            Path: 临时文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_filename = f"temp_{timestamp}{suffix}"
        temp_path = self.temp_dir / temp_filename
        
        # 创建空文件
        temp_path.touch()
        
        return temp_path
