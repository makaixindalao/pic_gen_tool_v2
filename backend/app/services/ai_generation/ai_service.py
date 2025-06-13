"""
AI生成服务
整合SD生成器和提示词构建器，提供统一的AI生成接口
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
from datetime import datetime
from PIL import Image
import json

from .sd_generator import SDGenerator
from .prompt_builder import PromptBuilder
from ..common.annotation_generator import AnnotationGenerator
from ..common.file_manager import FileManager
try:
    from ..common.yolo_detection import YOLODetectionService
except ImportError:
    # 如果真实YOLO服务不可用，使用模拟服务
    from ..common.mock_yolo_detection import YOLODetectionService
from ...core.config import get_settings

logger = logging.getLogger(__name__)

class AIGenerationService:
    """AI生成服务"""
    
    def __init__(self, model_path: str = None):
        """
        初始化AI生成服务
        
        Args:
            model_path: SD模型路径，如果为None则使用配置文件中的路径
        """
        settings = get_settings()
        
        # 使用配置文件中的模型路径
        if model_path is None:
            model_path = settings.get_ai_model_path()
        
        # 如果配置为使用本地模型，优先检查本地路径
        if settings.ai_use_local_model:
            # 尝试多个可能的路径
            possible_paths = [
                Path(settings.ai_local_models_dir) / "stable-diffusion-v1-5",
                Path.cwd() / settings.ai_local_models_dir / "stable-diffusion-v1-5",
                Path.cwd().parent / settings.ai_local_models_dir / "stable-diffusion-v1-5",  # 从backend目录向上一级
                Path(__file__).parent.parent.parent.parent / settings.ai_local_models_dir / "stable-diffusion-v1-5"
            ]
            
            for local_path in possible_paths:
                if local_path.exists() and local_path.is_dir():
                    model_path = str(local_path.resolve())
                    logger.info(f"使用本地模型路径: {model_path}")
                    break
            else:
                logger.warning(f"未找到本地模型，尝试的路径: {[str(p) for p in possible_paths]}")
        
        device = settings.get_ai_device()
        
        self.sd_generator = SDGenerator(model_path, device)
        self.prompt_builder = PromptBuilder()
        self.annotation_generator = AnnotationGenerator()
        self.file_manager = FileManager()
        self.yolo_service = YOLODetectionService()
        
        self.is_model_loaded = False
        self.generation_history = []
        
        logger.info(f"AI生成服务初始化完成，模型路径: {model_path}, 设备: {device}")
    
    async def initialize(self) -> bool:
        """
        异步初始化服务
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("开始初始化AI生成服务...")
            
            # 在线程池中加载模型以避免阻塞
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self.sd_generator.load_model)
            
            if success:
                self.is_model_loaded = True
                logger.info("AI生成服务初始化成功")
                return True
            else:
                logger.error("模型加载失败")
                return False
                
        except Exception as e:
            logger.error(f"AI生成服务初始化失败: {str(e)}")
            return False
    
    def build_prompt_from_selection(
        self,
        military_target: str,
        weather: str,
        scene: str,
        custom_prompt: str = "",
        style_strength: float = 0.7,
        technical_detail: float = 0.8
    ) -> Tuple[str, str]:
        """
        根据用户选择构建提示词
        
        Args:
            military_target: 军事目标
            weather: 天气条件
            scene: 场景环境
            custom_prompt: 自定义提示词
            style_strength: 风格强度
            technical_detail: 技术细节程度
            
        Returns:
            Tuple[str, str]: (正面提示词, 负面提示词)
        """
        if custom_prompt.strip():
            # 使用自定义提示词
            return self.prompt_builder.build_custom_prompt(
                base_prompt=custom_prompt,
                military_target=military_target,
                weather=weather,
                scene=scene,
                enhance_quality=True
            )
        else:
            # 自动构建提示词
            return self.prompt_builder.build_prompt(
                military_target=military_target,
                weather=weather,
                scene=scene,
                style_strength=style_strength,
                technical_detail=technical_detail,
                quality_boost=True
            )
    
    async def generate_images(
        self,
        military_target: str,
        weather: str,
        scene: str,
        num_images: int = 1,
        steps: int = 30,
        cfg_scale: float = 7.5,
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        scheduler_name: str = "DPM++ 2M Karras",
        custom_prompt: str = "",
        style_strength: float = 0.7,
        technical_detail: float = 0.8,
        save_images: bool = True,
        generate_annotations: bool = True,
        # YOLO检测参数
        enable_yolo_detection: bool = False,
        yolo_confidence: float = 0.5,
        yolo_nms_threshold: float = 0.4,
        yolo_save_original: bool = True,
        yolo_save_annotated: bool = True
    ) -> Dict[str, Any]:
        """
        生成图像
        
        Args:
            military_target: 军事目标
            weather: 天气条件
            scene: 场景环境
            num_images: 生成图像数量
            steps: 采样步数
            cfg_scale: CFG引导强度
            seed: 随机种子
            width: 图像宽度
            height: 图像高度
            scheduler_name: 采样器名称
            custom_prompt: 自定义提示词
            style_strength: 风格强度
            technical_detail: 技术细节程度
            save_images: 是否保存图像
            generate_annotations: 是否生成标注
            enable_yolo_detection: 是否启用YOLO检测
            yolo_confidence: YOLO检测置信度阈值
            yolo_nms_threshold: YOLO NMS阈值
            yolo_save_original: 是否保存原始图片
            yolo_save_annotated: 是否保存标注图片
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        if not self.is_model_loaded:
            raise RuntimeError("模型未加载，请先初始化服务")
        
        try:
            # 1. 构建提示词
            positive_prompt, negative_prompt = self.build_prompt_from_selection(
                military_target=military_target,
                weather=weather,
                scene=scene,
                custom_prompt=custom_prompt,
                style_strength=style_strength,
                technical_detail=technical_detail
            )
            
            logger.info(f"构建的提示词: {positive_prompt[:100]}...")
            
            # 2. 在线程池中生成图像
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                None,
                self.sd_generator.generate,
                positive_prompt,
                negative_prompt,
                num_images,
                steps,
                cfg_scale,
                seed,
                width,
                height,
                scheduler_name
            )
            
            # 3. 处理生成结果
            generation_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            result = {
                "generation_id": generation_id,
                "timestamp": timestamp.isoformat(),
                "device_used": self.sd_generator.device if self.sd_generator else "unknown",
                "parameters": {
                    "military_target": military_target,
                    "weather": weather,
                    "scene": scene,
                    "num_images": num_images,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "scheduler_name": scheduler_name,
                    "custom_prompt": custom_prompt,
                    "style_strength": style_strength,
                    "technical_detail": technical_detail
                },
                "prompts": {
                    "positive": positive_prompt,
                    "negative": negative_prompt
                },
                "images": [],
                "annotations": [],
                "yolo_results": None
            }
            
            # 4. 保存图像和生成标注
            yolo_results = None
            if enable_yolo_detection:
                # 初始化YOLO检测结果
                yolo_results = {
                    "total_detections": 0,
                    "detection_summary": {},
                    "annotated_images_saved": yolo_save_annotated,
                    "original_images_saved": yolo_save_original,
                    "images": []
                }
            
            if save_images or generate_annotations:
                for i, image in enumerate(images):
                    image_info = await self._process_generated_image(
                        image=image,
                        generation_id=generation_id,
                        image_index=i,
                        military_target=military_target,
                        weather=weather,
                        scene=scene,
                        save_image=save_images,
                        generate_annotation=generate_annotations,
                        # YOLO检测参数
                        enable_yolo_detection=enable_yolo_detection,
                        yolo_confidence=yolo_confidence,
                        yolo_nms_threshold=yolo_nms_threshold,
                        yolo_save_original=yolo_save_original,
                        yolo_save_annotated=yolo_save_annotated
                    )
                    result["images"].append(image_info)
                    
                    # 收集YOLO检测结果
                    if enable_yolo_detection and "yolo_detection" in image_info:
                        yolo_detection = image_info["yolo_detection"]
                        yolo_results["images"].append(yolo_detection)
                        yolo_results["total_detections"] += yolo_detection.get("total_detections", 0)
                        
                        # 合并检测统计
                        for target_type, count in yolo_detection.get("detection_summary", {}).items():
                            yolo_results["detection_summary"][target_type] = yolo_results["detection_summary"].get(target_type, 0) + count
            else:
                # 只返回图像对象
                result["images"] = images
            
            # 添加YOLO检测结果到返回值
            if enable_yolo_detection:
                result["yolo_results"] = yolo_results
            
            # 5. 记录生成历史
            self.generation_history.append({
                "generation_id": generation_id,
                "timestamp": timestamp,
                "parameters": result["parameters"],
                "num_images": len(images),
                "success": True
            })
            
            logger.info(f"成功生成 {len(images)} 张图像，ID: {generation_id}")
            return result
            
        except Exception as e:
            logger.error(f"图像生成失败: {str(e)}")
            
            # 记录失败历史
            self.generation_history.append({
                "generation_id": str(uuid.uuid4()),
                "timestamp": datetime.now(),
                "parameters": locals(),
                "num_images": 0,
                "success": False,
                "error": str(e)
            })
            
            raise
    
    async def _process_generated_image(
        self,
        image: Image.Image,
        generation_id: str,
        image_index: int,
        military_target: str,
        weather: str,
        scene: str,
        save_image: bool = True,
        generate_annotation: bool = True,
        # YOLO检测参数
        enable_yolo_detection: bool = False,
        yolo_confidence: float = 0.5,
        yolo_nms_threshold: float = 0.4,
        yolo_save_original: bool = True,
        yolo_save_annotated: bool = True
    ) -> Dict[str, Any]:
        """
        处理生成的图像
        
        Args:
            image: 生成的图像
            generation_id: 生成ID
            image_index: 图像索引
            military_target: 军事目标
            weather: 天气条件
            scene: 场景环境
            save_image: 是否保存图像
            generate_annotation: 是否生成标注
            
        Returns:
            Dict[str, Any]: 图像信息
        """
        image_info = {
            "index": image_index,
            "width": image.width,
            "height": image.height,
            "format": image.format or "PNG"
        }
        
        if save_image:
            # 保存图像
            filename = f"{generation_id}_{image_index:03d}.png"
            
            # 将PIL Image转换为字节数据
            import io
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_data = image_bytes.getvalue()
            
            # 准备元数据
            metadata = {
                "military_target": military_target,
                "weather": weather,
                "scene": scene,
                "generation_id": generation_id,
                "image_index": image_index,
                "width": image.width,
                "height": image.height
            }
            
            image_path, file_info = self.file_manager.save_generated_image(
                image_data=image_data,
                filename=filename,
                generation_type="ai_generated",
                metadata=metadata
            )
            # 确保返回绝对路径
            abs_image_path = os.path.abspath(str(image_path))
            image_info["file_path"] = abs_image_path
            image_info["filename"] = filename
            image_info["file_info"] = file_info
        
        if generate_annotation:
            # 生成标注
            annotation = await self._generate_annotation_for_image(
                image=image,
                military_target=military_target,
                weather=weather,
                scene=scene
            )
            image_info["annotation"] = annotation
            
            if save_image and "file_path" in image_info:
                # 保存标注文件
                annotation_path = self.file_manager.save_annotation(
                    annotation=annotation,
                    image_path=image_info["file_path"]
                )
                image_info["annotation_path"] = str(annotation_path)
        
        # YOLO检测处理
        if enable_yolo_detection and save_image and "file_path" in image_info:
            try:
                yolo_detection = await self._perform_yolo_detection(
                    image_path=image_info["file_path"],
                    generation_id=generation_id,
                    image_index=image_index,
                    confidence_threshold=yolo_confidence,
                    nms_threshold=yolo_nms_threshold,
                    save_original=yolo_save_original,
                    save_annotated=yolo_save_annotated
                )
                image_info["yolo_detection"] = yolo_detection
                logger.info(f"YOLO检测完成，检测到 {yolo_detection.get('total_detections', 0)} 个目标")
            except Exception as e:
                logger.error(f"YOLO检测失败: {str(e)}")
                image_info["yolo_detection"] = {
                    "error": str(e),
                    "total_detections": 0,
                    "detection_summary": {}
                }
        
        return image_info
    
    async def _generate_annotation_for_image(
        self,
        image: Image.Image,
        military_target: str,
        weather: str,
        scene: str
    ) -> Dict[str, Any]:
        """
        为生成的图像创建标注
        
        Args:
            image: 图像对象
            military_target: 军事目标
            weather: 天气条件
            scene: 场景环境
            
        Returns:
            Dict[str, Any]: COCO格式标注
        """
        # 创建基础标注信息
        image_width, image_height = image.size
        
        # 估算目标位置（简化版本，实际应用中可能需要目标检测）
        # 这里假设目标位于图像中心区域
        center_x = image_width // 2
        center_y = image_height // 2
        
        # 根据目标类型估算边界框大小
        target_size_ratios = {
            "坦克": (0.3, 0.2),  # 宽度比例, 高度比例
            "战机": (0.4, 0.25),
            "舰艇": (0.5, 0.3)
        }
        
        width_ratio, height_ratio = target_size_ratios.get(military_target, (0.3, 0.2))
        bbox_width = int(image_width * width_ratio)
        bbox_height = int(image_height * height_ratio)
        
        bbox_x = center_x - bbox_width // 2
        bbox_y = center_y - bbox_height // 2
        
        # 确保边界框在图像范围内
        bbox_x = max(0, min(bbox_x, image_width - bbox_width))
        bbox_y = max(0, min(bbox_y, image_height - bbox_height))
        
        # 创建COCO格式标注
        annotation = self.annotation_generator.create_coco_annotation(
            image_id=1,
            category_name=military_target,
            bbox=[bbox_x, bbox_y, bbox_width, bbox_height],
            image_width=image_width,
            image_height=image_height,
            additional_info={
                "weather": weather,
                "scene": scene,
                "generation_method": "ai_generated",
                "confidence": 0.95  # AI生成的图像假设有高置信度
            }
        )
        
        return annotation
    
    async def _perform_yolo_detection(
        self,
        image_path: str,
        generation_id: str,
        image_index: int,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        save_original: bool = True,
        save_annotated: bool = True
    ) -> Dict[str, Any]:
        """
        执行YOLO检测
        
        Args:
            image_path: 图像文件路径
            generation_id: 生成ID
            image_index: 图像索引
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            save_original: 是否保存原始图片
            save_annotated: 是否保存标注图片
            
        Returns:
            Dict[str, Any]: YOLO检测结果
        """
        try:
            # 确保YOLO模型已加载
            if not self.yolo_service.is_loaded:
                logger.info("正在加载YOLO模型...")
                success = self.yolo_service.load_model()
                if not success:
                    raise RuntimeError("YOLO模型加载失败")
            
            # 创建YOLO输出目录
            yolo_output_dir = os.path.join(
                os.path.dirname(image_path),
                "yolo_detection"
            )
            os.makedirs(yolo_output_dir, exist_ok=True)
            
            # 在线程池中执行YOLO检测以避免阻塞
            loop = asyncio.get_event_loop()
            yolo_result = await loop.run_in_executor(
                None,
                self.yolo_service.process_image,
                image_path,
                confidence_threshold,
                nms_threshold,
                save_original,
                save_annotated,
                yolo_output_dir
            )
            
            # 添加生成相关信息
            yolo_result.update({
                "generation_id": generation_id,
                "image_index": image_index,
                "yolo_model_info": self.yolo_service.get_model_info()
            })
            
            return yolo_result
            
        except Exception as e:
            logger.error(f"YOLO检测执行失败: {str(e)}")
            raise
    
    def get_prompt_suggestions(self, military_target: str) -> Dict[str, List[str]]:
        """
        获取提示词建议
        
        Args:
            military_target: 军事目标类型
            
        Returns:
            Dict[str, List[str]]: 提示词建议
        """
        return self.prompt_builder.get_prompt_suggestions(military_target)
    
    def optimize_prompt(self, prompt: str) -> str:
        """
        优化提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 优化后的提示词
        """
        return self.prompt_builder.optimize_prompt(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        info = self.sd_generator.get_model_info()
        info.update({
            "service_initialized": self.is_model_loaded,
            "generation_count": len(self.generation_history),
            "available_targets": list(self.prompt_builder.military_targets.keys()),
            "available_weather": list(self.prompt_builder.weather_conditions.keys()),
            "available_scenes": list(self.prompt_builder.scene_environments.keys())
        })
        return info
    
    def get_generation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取生成历史
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            List[Dict[str, Any]]: 生成历史记录
        """
        return self.generation_history[-limit:]
    
    def clear_generation_history(self):
        """清空生成历史"""
        self.generation_history.clear()
        logger.info("生成历史已清空")
    
    async def batch_generate(
        self,
        generation_configs: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        批量生成图像
        
        Args:
            generation_configs: 生成配置列表
            progress_callback: 进度回调函数
            
        Returns:
            List[Dict[str, Any]]: 批量生成结果
        """
        results = []
        total_configs = len(generation_configs)
        
        for i, config in enumerate(generation_configs):
            try:
                logger.info(f"批量生成进度: {i+1}/{total_configs}")
                
                result = await self.generate_images(**config)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total_configs, result)
                    
            except Exception as e:
                logger.error(f"批量生成第 {i+1} 项失败: {str(e)}")
                results.append({
                    "error": str(e),
                    "config": config,
                    "success": False
                })
        
        logger.info(f"批量生成完成，成功: {sum(1 for r in results if r.get('success', True))}/{total_configs}")
        return results
    
    def cleanup(self):
        """清理资源"""
        if self.sd_generator:
            self.sd_generator.unload_model()
        self.is_model_loaded = False
        logger.info("AI生成服务资源已清理") 