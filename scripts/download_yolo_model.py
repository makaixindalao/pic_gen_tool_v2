#!/usr/bin/env python3
"""
YOLO模型下载脚本
手动下载YOLO模型文件
"""

import os
import requests
from pathlib import Path
import sys

def download_yolo_model():
    """下载YOLO模型"""
    try:
        print("=== YOLO模型下载工具 ===")
        
        # 创建模型目录
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # YOLO模型下载链接
        model_urls = {
            "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        }
        
        # 选择要下载的模型
        print("可用的YOLO模型:")
        print("1. YOLOv8n (轻量级, ~6MB)")
        print("2. YOLOv8s (标准版, ~22MB)")
        
        choice = input("请选择要下载的模型 (1-2, 默认1): ").strip()
        if choice == "2":
            model_name = "yolov8s.pt"
        else:
            model_name = "yolov8n.pt"
        
        model_path = models_dir / model_name
        model_url = model_urls[model_name]
        
        # 检查模型是否已存在
        if model_path.exists():
            print(f"模型文件已存在: {model_path}")
            overwrite = input("是否重新下载? (y/N): ").strip().lower()
            if overwrite != 'y':
                print("跳过下载")
                return str(model_path)
        
        print(f"正在下载 {model_name}...")
        print(f"下载地址: {model_url}")
        
        # 下载模型
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # 显示进度
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r下载进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        print(f"\n✓ 模型下载完成: {model_path}")
        print(f"文件大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return str(model_path)
        
    except requests.exceptions.RequestException as e:
        print(f"✗ 网络下载失败: {str(e)}")
        print("\n备选方案:")
        print("1. 检查网络连接")
        print("2. 使用代理或VPN")
        print("3. 手动下载模型文件到 models/ 目录")
        return None
    except Exception as e:
        print(f"✗ 下载失败: {str(e)}")
        return None

def create_mock_yolo_service():
    """创建模拟YOLO服务（用于测试）"""
    print("\n=== 创建模拟YOLO服务 ===")
    
    mock_service_path = Path("backend/app/services/common/mock_yolo_detection.py")
    
    mock_service_code = '''"""
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
'''
    
    with open(mock_service_path, 'w', encoding='utf-8') as f:
        f.write(mock_service_code)
    
    print(f"✓ 模拟YOLO服务已创建: {mock_service_path}")
    print("现在可以使用模拟服务进行YOLO功能测试")
    
    return str(mock_service_path)

def main():
    """主函数"""
    print("YOLO模型下载工具")
    print("1. 尝试下载真实YOLO模型")
    print("2. 创建模拟YOLO服务（用于测试）")
    
    choice = input("请选择操作 (1-2, 默认2): ").strip()
    
    if choice == "1":
        model_path = download_yolo_model()
        if model_path:
            print(f"\n✅ YOLO模型下载成功: {model_path}")
            print("现在可以使用真实的YOLO检测功能")
        else:
            print("\n❌ YOLO模型下载失败")
            print("建议使用模拟服务进行测试")
    else:
        mock_path = create_mock_yolo_service()
        print(f"\n✅ 模拟YOLO服务创建成功")
        print("现在可以使用模拟服务测试YOLO功能集成")

if __name__ == "__main__":
    main() 