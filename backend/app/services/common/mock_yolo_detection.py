"""
模拟YOLO检测服务
用于在没有真实YOLO模型时进行测试
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import logging
import random

logger = logging.getLogger(__name__)

class MockYOLODetectionService:
    """模拟YOLO目标检测服务"""
    
    def __init__(self):
        """初始化模拟YOLO检测服务"""
        self.model = None
        self.is_loaded = True  # 模拟服务总是可用
        self.device = "cpu"
        
        # 军事目标类别映射
        self.military_class_mapping = {
            2: "坦克",
            5: "战机", 
            8: "舰艇",
        }
        
        # 颜色映射
        self.colors = {
            "坦克": (0, 255, 0),
            "战机": (255, 0, 0),
            "舰艇": (0, 0, 255),
        }
    
    def load_model(self, model_path: str = "yolov8n.pt") -> bool:
        """模拟加载YOLO模型"""
        logger.info(f"模拟加载YOLO模型: {model_path}")
        self.is_loaded = True
        return True
    
    def detect_objects(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """模拟检测图像中的目标"""
        if not self.is_loaded:
            return []
        
        height, width = image.shape[:2]
        detections = []
        
        # 随机生成1-3个检测结果
        num_detections = random.randint(1, 3)
        
        for i in range(num_detections):
            # 随机选择目标类型
            class_id = random.choice(list(self.military_class_mapping.keys()))
            class_name = self.military_class_mapping[class_id]
            
            # 随机生成边界框
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            x2 = random.randint(x1 + 50, min(x1 + 200, width))
            y2 = random.randint(y1 + 30, min(y1 + 150, height))
            
            # 随机置信度
            confidence = random.uniform(confidence_threshold, 1.0)
            
            detection = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": class_name,
                "area": float((x2 - x1) * (y2 - y1))
            }
            detections.append(detection)
        
        logger.info(f"模拟检测到 {len(detections)} 个军事目标")
        return detections
    
    def draw_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """在图像上绘制检测结果"""
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 获取颜色
            color = self.colors.get(class_name, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_image
    
    def process_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        save_original: bool = True,
        save_annotated: bool = True,
        output_dir: str = "yolo_results"
    ) -> Dict[str, Any]:
        """处理单张图像"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 执行检测
            detections = self.detect_objects(
                image, confidence_threshold, nms_threshold
            )
            
            result = {
                "image_path": image_path,
                "detections": detections,
                "total_detections": len(detections),
                "detection_summary": {},
                "original_image_path": None,
                "annotated_image_path": None
            }
            
            # 统计检测结果
            for detection in detections:
                class_name = detection["class_name"]
                result["detection_summary"][class_name] = result["detection_summary"].get(class_name, 0) + 1
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 保存原始图像
            if save_original:
                original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
                cv2.imwrite(original_path, image)
                result["original_image_path"] = original_path
                logger.info(f"保存原始图像: {original_path}")
            
            # 保存标注图像
            if save_annotated and detections:
                annotated_image = self.draw_annotations(image, detections)
                annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
                cv2.imwrite(annotated_path, annotated_image)
                result["annotated_image_path"] = annotated_path
                logger.info(f"保存标注图像: {annotated_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            return {
                "image_path": image_path,
                "detections": [],
                "total_detections": 0,
                "detection_summary": {},
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "is_loaded": self.is_loaded,
            "device": self.device,
            "supported_classes": list(self.military_class_mapping.values()),
            "class_mapping": self.military_class_mapping,
            "model_type": "mock"
        }

# 为了兼容性，创建别名
YOLODetectionService = MockYOLODetectionService
