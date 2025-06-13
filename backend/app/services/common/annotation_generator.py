"""
标注生成器
用于生成COCO格式的标注文件
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

class AnnotationGenerator:
    """COCO格式标注生成器"""
    
    def __init__(self):
        """初始化标注生成器"""
        self.categories = {
            "坦克": {"id": 1, "name": "tank", "supercategory": "military_vehicle"},
            "战机": {"id": 2, "name": "fighter_jet", "supercategory": "military_aircraft"},
            "舰艇": {"id": 3, "name": "warship", "supercategory": "military_vessel"}
        }
        
        self.annotation_id = 1
        self.image_id = 1
    
    def create_coco_dataset(
        self,
        dataset_name: str = "Military Target Dataset",
        description: str = "Generated military target dataset",
        version: str = "1.0"
    ) -> Dict[str, Any]:
        """
        创建COCO格式数据集结构
        
        Args:
            dataset_name: 数据集名称
            description: 数据集描述
            version: 版本号
            
        Returns:
            Dict: COCO格式数据集结构
        """
        return {
            "info": {
                "description": description,
                "url": "",
                "version": version,
                "year": datetime.now().year,
                "contributor": "Military Target Generation Platform",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": cat_info["id"],
                    "name": cat_info["name"],
                    "supercategory": cat_info["supercategory"]
                }
                for cat_info in self.categories.values()
            ]
        }
    
    def add_image_annotation(
        self,
        coco_dataset: Dict[str, Any],
        image_path: str,
        target_type: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        segmentation: Optional[List[List[float]]] = None
    ) -> bool:
        """
        添加图像和标注信息
        
        Args:
            coco_dataset: COCO数据集结构
            image_path: 图像文件路径
            target_type: 军事目标类型
            bbox: 边界框 (x, y, width, height)
            segmentation: 分割掩码
            
        Returns:
            bool: 是否成功添加
        """
        try:
            # 检查图像文件
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                return False
            
            # 获取图像信息
            with Image.open(image_path) as img:
                width, height = img.size
            
            # 添加图像信息
            image_info = {
                "id": self.image_id,
                "width": width,
                "height": height,
                "file_name": os.path.basename(image_path),
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().isoformat()
            }
            coco_dataset["images"].append(image_info)
            
            # 添加标注信息
            if target_type in self.categories:
                category_id = self.categories[target_type]["id"]
                
                # 如果没有提供边界框，使用整个图像
                if bbox is None:
                    bbox = (0, 0, width, height)
                
                # 如果没有提供分割掩码，使用边界框创建简单掩码
                if segmentation is None:
                    x, y, w, h = bbox
                    segmentation = [[
                        x, y,
                        x + w, y,
                        x + w, y + h,
                        x, y + h
                    ]]
                
                annotation = {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": bbox[2] * bbox[3],
                    "bbox": list(bbox),
                    "iscrowd": 0
                }
                coco_dataset["annotations"].append(annotation)
                
                self.annotation_id += 1
            
            self.image_id += 1
            return True
            
        except Exception as e:
            print(f"添加图像标注失败: {str(e)}")
            return False
    
    def generate_auto_annotation(
        self,
        image_path: str,
        target_type: str
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[List[List[float]]]]:
        """
        自动生成标注信息
        
        Args:
            image_path: 图像路径
            target_type: 目标类型
            
        Returns:
            Tuple: (边界框, 分割掩码)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            
            # 简单的自动标注：假设目标在图像中央区域
            center_x, center_y = width // 2, height // 2
            
            # 根据目标类型调整边界框大小
            if target_type == "坦克":
                w, h = min(width * 0.6, 400), min(height * 0.4, 200)
            elif target_type == "战机":
                w, h = min(width * 0.7, 500), min(height * 0.3, 150)
            elif target_type == "舰艇":
                w, h = min(width * 0.8, 600), min(height * 0.5, 300)
            else:
                w, h = width * 0.5, height * 0.5
            
            x = max(0, center_x - w // 2)
            y = max(0, center_y - h // 2)
            
            # 确保边界框在图像范围内
            w = min(w, width - x)
            h = min(h, height - y)
            
            bbox = (int(x), int(y), int(w), int(h))
            
            # 生成简单的分割掩码（矩形）
            segmentation = [[
                x, y,
                x + w, y,
                x + w, y + h,
                x, y + h
            ]]
            
            return bbox, segmentation
            
        except Exception as e:
            print(f"自动生成标注失败: {str(e)}")
            return None, None
    
    def create_coco_annotation(
        self,
        image_id: int,
        category_name: str,
        bbox: List[int],
        image_width: int,
        image_height: int,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建单个COCO格式标注
        
        Args:
            image_id: 图像ID
            category_name: 类别名称
            bbox: 边界框 [x, y, width, height]
            image_width: 图像宽度
            image_height: 图像高度
            additional_info: 额外信息
            
        Returns:
            Dict: COCO格式标注
        """
        if category_name not in self.categories:
            raise ValueError(f"未知的类别: {category_name}")
        
        category_id = self.categories[category_name]["id"]
        x, y, w, h = bbox
        
        # 确保边界框在图像范围内
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        w = max(1, min(w, image_width - x))
        h = max(1, min(h, image_height - y))
        
        # 创建分割掩码（简单矩形）
        segmentation = [[
            x, y,
            x + w, y,
            x + w, y + h,
            x, y + h
        ]]
        
        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": w * h,
            "bbox": [x, y, w, h],
            "iscrowd": 0
        }
        
        # 添加额外信息
        if additional_info:
            annotation.update(additional_info)
        
        self.annotation_id += 1
        return annotation
    
    def save_annotations(
        self,
        coco_dataset: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        保存标注文件
        
        Args:
            coco_dataset: COCO数据集
            output_path: 输出文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_dataset, f, indent=2, ensure_ascii=False)
            
            print(f"标注文件已保存: {output_path}")
            return True
            
        except Exception as e:
            print(f"保存标注文件失败: {str(e)}")
            return False
    
    def validate_annotations(self, coco_dataset: Dict[str, Any]) -> bool:
        """
        验证标注数据的完整性
        
        Args:
            coco_dataset: COCO数据集
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查必要字段
            required_fields = ["info", "images", "annotations", "categories"]
            for field in required_fields:
                if field not in coco_dataset:
                    print(f"缺少必要字段: {field}")
                    return False
            
            # 检查图像和标注数量
            num_images = len(coco_dataset["images"])
            num_annotations = len(coco_dataset["annotations"])
            
            print(f"数据集验证通过: {num_images} 张图像, {num_annotations} 个标注")
            return True
            
        except Exception as e:
            print(f"验证标注数据失败: {str(e)}")
            return False
    
    def reset_counters(self):
        """重置ID计数器"""
        self.annotation_id = 1
        self.image_id = 1
