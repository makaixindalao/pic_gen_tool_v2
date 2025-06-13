"""
YOLO目标检测服务
用于检测生成图像中的军事目标并生成标注
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class YOLODetectionService:
    """YOLO目标检测服务"""
    
    def __init__(self):
        """初始化YOLO检测服务"""
        self.model = None
        self.is_loaded = False
        self.device = "cpu"  # 默认使用CPU
        
        # 军事目标类别映射 (COCO类别ID -> 军事目标名称)
        self.military_class_mapping = {
            2: "坦克",      # car -> 坦克 (近似映射)
            5: "战机",      # airplane -> 战机
            8: "舰艇",      # boat -> 舰艇
            # 可以根据实际需要扩展更多映射
        }
        
        # 颜色映射
        self.colors = {
            "坦克": (0, 255, 0),    # 绿色
            "战机": (255, 0, 0),    # 红色
            "舰艇": (0, 0, 255),    # 蓝色
        }
    
    def load_model(self, model_path: str = "yolov8n.pt") -> bool:
        """
        加载YOLO模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 尝试导入ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("未安装ultralytics库，请运行: pip install ultralytics")
                logger.info("将使用模拟YOLO服务")
                return self._fallback_to_mock()
            
            # 检查依赖
            try:
                import torch
                import torchvision
                if torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info("使用GPU进行YOLO检测")
                else:
                    self.device = "cpu"
                    logger.info("使用CPU进行YOLO检测")
            except ImportError as e:
                logger.error(f"缺少依赖: {str(e)}")
                logger.info("将使用模拟YOLO服务")
                return self._fallback_to_mock()
            
            # 加载模型
            logger.info(f"正在加载YOLO模型: {model_path}")
            self.model = YOLO(model_path)
            
            # 将模型移动到指定设备
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            self.is_loaded = True
            logger.info("YOLO模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"YOLO模型加载失败: {str(e)}")
            logger.info("将使用模拟YOLO服务")
            return self._fallback_to_mock()
    
    def _fallback_to_mock(self) -> bool:
        """回退到模拟服务"""
        try:
            from .mock_yolo_detection import MockYOLODetectionService
            
            # 替换当前实例的方法
            mock_service = MockYOLODetectionService()
            self.detect_objects = mock_service.detect_objects
            self.draw_annotations = mock_service.draw_annotations
            self.process_image = mock_service.process_image
            self.get_model_info = mock_service.get_model_info
            
            self.is_loaded = True
            self.device = "cpu"
            logger.info("已切换到模拟YOLO服务")
            return True
        except Exception as e:
            logger.error(f"模拟YOLO服务也不可用: {str(e)}")
            self.is_loaded = False
            return False
    
    def detect_objects(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像 (numpy数组)
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            List[Dict]: 检测结果列表
        """
        if not self.is_loaded:
            logger.warning("YOLO模型未加载")
            return []
        
        try:
            # 执行检测
            results = self.model(
                image,
                conf=confidence_threshold,
                iou=nms_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            # 处理检测结果
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                    confidences = result.boxes.conf.cpu().numpy()  # 置信度
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # 类别ID
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # 检查是否为军事目标类别
                        if class_id in self.military_class_mapping:
                            x1, y1, x2, y2 = box
                            
                            detection = {
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": float(conf),
                                "class_id": int(class_id),
                                "class_name": self.military_class_mapping[class_id],
                                "area": float((x2 - x1) * (y2 - y1))
                            }
                            detections.append(detection)
            
            logger.info(f"检测到 {len(detections)} 个军事目标")
            return detections
            
        except Exception as e:
            logger.error(f"目标检测失败: {str(e)}")
            return []
    
    def draw_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            
        Returns:
            np.ndarray: 标注后的图像
        """
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
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            save_original: 是否保存原始图像
            save_annotated: 是否保存标注图像
            output_dir: 输出目录
            
        Returns:
            Dict: 处理结果
        """
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
    
    def convert_to_coco_format(
        self,
        detections: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        image_id: int = 1
    ) -> List[Dict[str, Any]]:
        """
        将检测结果转换为COCO格式
        
        Args:
            detections: 检测结果
            image_width: 图像宽度
            image_height: 图像高度
            image_id: 图像ID
            
        Returns:
            List[Dict]: COCO格式标注
        """
        coco_annotations = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            width = x2 - x1
            height = y2 - y1
            
            # COCO格式: [x, y, width, height]
            bbox_coco = [x1, y1, width, height]
            
            # 创建分割掩码 (简单矩形)
            segmentation = [[
                x1, y1,
                x2, y1,
                x2, y2,
                x1, y2
            ]]
            
            annotation = {
                "id": i + 1,
                "image_id": image_id,
                "category_id": detection["class_id"],
                "segmentation": segmentation,
                "area": detection["area"],
                "bbox": bbox_coco,
                "iscrowd": 0,
                "confidence": detection["confidence"],
                "class_name": detection["class_name"]
            }
            
            coco_annotations.append(annotation)
        
        return coco_annotations
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        return {
            "is_loaded": self.is_loaded,
            "device": self.device,
            "supported_classes": list(self.military_class_mapping.values()),
            "class_mapping": self.military_class_mapping
        } 