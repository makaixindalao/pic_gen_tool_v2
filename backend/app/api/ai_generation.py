"""
AI生成相关的API路由
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import asyncio

from ..services.ai_generation.ai_service import AIGenerationService
from ..core.dependencies import get_ai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI生成"])

# Pydantic模型定义
class GenerationRequest(BaseModel):
    """图像生成请求"""
    military_target: str = Field(..., description="军事目标类型")
    weather: str = Field(..., description="天气条件")
    scene: str = Field(..., description="场景环境")
    num_images: int = Field(1, ge=1, le=10, description="生成图像数量")
    steps: int = Field(30, ge=10, le=100, description="采样步数")
    cfg_scale: float = Field(7.5, ge=1.0, le=20.0, description="CFG引导强度")
    seed: int = Field(-1, description="随机种子，-1表示随机")
    width: int = Field(512, ge=256, le=1024, description="图像宽度")
    height: int = Field(512, ge=256, le=1024, description="图像高度")
    scheduler_name: str = Field("DPM++ 2M Karras", description="采样器名称")
    custom_prompt: str = Field("", description="自定义提示词")
    style_strength: float = Field(0.7, ge=0.0, le=1.0, description="风格强度")
    technical_detail: float = Field(0.8, ge=0.0, le=1.0, description="技术细节程度")
    save_images: bool = Field(True, description="是否保存图像")
    generate_annotations: bool = Field(True, description="是否生成标注")
    # YOLO检测参数
    enable_yolo_detection: bool = Field(False, description="是否启用YOLO检测")
    yolo_confidence: float = Field(0.5, ge=0.1, le=1.0, description="YOLO检测置信度阈值")
    yolo_nms_threshold: float = Field(0.4, ge=0.1, le=1.0, description="YOLO NMS阈值")
    yolo_save_original: bool = Field(True, description="是否保存原始图片")
    yolo_save_annotated: bool = Field(True, description="是否保存标注图片")

class PromptBuildRequest(BaseModel):
    """提示词构建请求"""
    military_target: str = Field(..., description="军事目标类型")
    weather: str = Field(..., description="天气条件")
    scene: str = Field(..., description="场景环境")
    custom_prompt: str = Field("", description="自定义提示词")
    style_strength: float = Field(0.7, ge=0.0, le=1.0, description="风格强度")
    technical_detail: float = Field(0.8, ge=0.0, le=1.0, description="技术细节程度")

class BatchGenerationRequest(BaseModel):
    """批量生成请求"""
    configs: List[GenerationRequest] = Field(..., description="生成配置列表")

class GenerationResponse(BaseModel):
    """生成响应"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# API路由
@router.post("/generate", response_model=GenerationResponse)
async def generate_images(
    request: GenerationRequest,
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    生成图像
    """
    try:
        logger.info(f"收到图像生成请求: {request.military_target}, {request.weather}, {request.scene}")
        
        # 检查AI服务是否可用
        if not ai_service.is_model_loaded:
            logger.warning("AI模型未加载，尝试重新初始化...")
            try:
                await ai_service.initialize()
            except Exception as e:
                logger.error(f"AI模型初始化失败: {str(e)}")
                raise HTTPException(
                    status_code=503, 
                    detail="AI生成服务暂时不可用，请检查模型配置或使用传统生成模式"
                )
        
        result = await ai_service.generate_images(
            military_target=request.military_target,
            weather=request.weather,
            scene=request.scene,
            num_images=request.num_images,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            width=request.width,
            height=request.height,
            scheduler_name=request.scheduler_name,
            custom_prompt=request.custom_prompt,
            style_strength=request.style_strength,
            technical_detail=request.technical_detail,
            save_images=request.save_images,
            generate_annotations=request.generate_annotations,
            # YOLO检测参数
            enable_yolo_detection=request.enable_yolo_detection,
            yolo_confidence=request.yolo_confidence,
            yolo_nms_threshold=request.yolo_nms_threshold,
            yolo_save_original=request.yolo_save_original,
            yolo_save_annotated=request.yolo_save_annotated
        )
        
        return GenerationResponse(
            success=True,
            message=f"成功生成 {len(result['images'])} 张图像",
            data=result
        )
        
    except Exception as e:
        logger.error(f"图像生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图像生成失败: {str(e)}")

@router.post("/build-prompt")
async def build_prompt(
    request: PromptBuildRequest,
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    构建提示词
    """
    try:
        positive_prompt, negative_prompt = ai_service.build_prompt_from_selection(
            military_target=request.military_target,
            weather=request.weather,
            scene=request.scene,
            custom_prompt=request.custom_prompt,
            style_strength=request.style_strength,
            technical_detail=request.technical_detail
        )
        
        return {
            "success": True,
            "data": {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt
            }
        }
        
    except Exception as e:
        logger.error(f"提示词构建失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提示词构建失败: {str(e)}")

@router.get("/prompt-suggestions/{military_target}")
async def get_prompt_suggestions(
    military_target: str,
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    获取提示词建议
    """
    try:
        suggestions = ai_service.get_prompt_suggestions(military_target)
        
        return {
            "success": True,
            "data": suggestions
        }
        
    except Exception as e:
        logger.error(f"获取提示词建议失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取提示词建议失败: {str(e)}")

@router.post("/optimize-prompt")
async def optimize_prompt(
    prompt: str,
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    优化提示词
    """
    try:
        optimized_prompt = ai_service.optimize_prompt(prompt)
        
        return {
            "success": True,
            "data": {
                "original_prompt": prompt,
                "optimized_prompt": optimized_prompt
            }
        }
        
    except Exception as e:
        logger.error(f"提示词优化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提示词优化失败: {str(e)}")

@router.post("/batch-generate", response_model=GenerationResponse)
async def batch_generate(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    批量生成图像
    """
    try:
        logger.info(f"收到批量生成请求，共 {len(request.configs)} 个配置")
        
        # 将批量生成任务添加到后台任务
        task_id = f"batch_{len(request.configs)}_{asyncio.get_event_loop().time()}"
        
        # 这里可以集成Celery来处理长时间运行的任务
        # 现在先直接执行
        results = await ai_service.batch_generate(
            generation_configs=[config.dict() for config in request.configs]
        )
        
        return GenerationResponse(
            success=True,
            message=f"批量生成完成，共处理 {len(results)} 个任务",
            data={
                "task_id": task_id,
                "results": results,
                "total_configs": len(request.configs),
                "successful_count": sum(1 for r in results if r.get('success', True))
            }
        )
        
    except Exception as e:
        logger.error(f"批量生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")

@router.get("/model-info")
async def get_model_info(
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    获取模型信息
    """
    try:
        info = ai_service.get_model_info()
        
        return {
            "success": True,
            "data": info
        }
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")

@router.get("/generation-history")
async def get_generation_history(
    limit: int = 50,
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    获取生成历史
    """
    try:
        history = ai_service.get_generation_history(limit=limit)
        
        return {
            "success": True,
            "data": {
                "history": history,
                "total_count": len(history)
            }
        }
        
    except Exception as e:
        logger.error(f"获取生成历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取生成历史失败: {str(e)}")

@router.delete("/generation-history")
async def clear_generation_history(
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    清空生成历史
    """
    try:
        ai_service.clear_generation_history()
        
        return {
            "success": True,
            "message": "生成历史已清空"
        }
        
    except Exception as e:
        logger.error(f"清空生成历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空生成历史失败: {str(e)}")

@router.post("/initialize")
async def initialize_service(
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    初始化AI生成服务
    """
    try:
        logger.info("开始初始化AI生成服务...")
        
        success = await ai_service.initialize()
        
        if success:
            return {
                "success": True,
                "message": "AI生成服务初始化成功"
            }
        else:
            return {
                "success": False,
                "message": "AI生成服务初始化失败，请检查模型配置"
            }
            
    except Exception as e:
        logger.error(f"AI生成服务初始化失败: {str(e)}")
        return {
            "success": False,
            "message": f"AI生成服务初始化失败: {str(e)}"
        }

@router.get("/status")
async def get_service_status(
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    获取服务状态
    """
    try:
        info = ai_service.get_model_info()
        
        return {
            "success": True,
            "data": {
                "service_status": "running" if info.get("service_initialized") else "stopped",
                "model_loaded": info.get("is_loaded", False),
                "device": info.get("device"),
                "current_scheduler": info.get("current_scheduler"),
                "generation_count": info.get("generation_count", 0),
                "available_options": {
                    "targets": info.get("available_targets", []),
                    "weather": info.get("available_weather", []),
                    "scenes": info.get("available_scenes", []),
                    "schedulers": info.get("available_schedulers", [])
                }
            }
        }
        
    except Exception as e:
        logger.error(f"获取服务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取服务状态失败: {str(e)}")

@router.get("/download/{generation_id}/{image_index}")
async def download_generated_image(
    generation_id: str,
    image_index: int,
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    下载生成的图像
    """
    try:
        # 这里需要根据generation_id和image_index找到对应的文件
        # 简化实现，实际应该从数据库或文件系统中查找
        filename = f"{generation_id}_{image_index:03d}.png"
        file_path = f"data/generated/ai_generated/{filename}"
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="image/png"
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="图像文件未找到")
    except Exception as e:
        logger.error(f"下载图像失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"下载图像失败: {str(e)}")

# 预设配置相关的路由
@router.get("/presets")
async def get_generation_presets():
    """
    获取生成预设配置
    """
    presets = {
        "高质量": {
            "steps": 50,
            "cfg_scale": 8.0,
            "scheduler_name": "DPM++ 2M Karras",
            "style_strength": 0.8,
            "technical_detail": 0.9
        },
        "快速生成": {
            "steps": 20,
            "cfg_scale": 7.0,
            "scheduler_name": "Euler a",
            "style_strength": 0.6,
            "technical_detail": 0.7
        },
        "平衡模式": {
            "steps": 30,
            "cfg_scale": 7.5,
            "scheduler_name": "DPM++ 2M Karras",
            "style_strength": 0.7,
            "technical_detail": 0.8
        },
        "艺术风格": {
            "steps": 40,
            "cfg_scale": 9.0,
            "scheduler_name": "Heun",
            "style_strength": 0.9,
            "technical_detail": 0.6
        }
    }
    
    return {
        "success": True,
        "data": presets
    }

@router.get("/schedulers")
async def get_available_schedulers(
    ai_service: AIGenerationService = Depends(get_ai_service)
):
    """
    获取可用的采样器列表
    """
    try:
        info = ai_service.get_model_info()
        schedulers = info.get("available_schedulers", [])
        
        return {
            "success": True,
            "data": {
                "schedulers": schedulers,
                "current": info.get("current_scheduler"),
                "descriptions": {
                    "DPM++ 2M Karras": "高质量，适合大多数场景",
                    "Euler a": "快速生成，适合预览",
                    "Euler": "稳定生成，适合批量处理",
                    "LMS": "经典算法，兼容性好",
                    "Heun": "高精度，适合艺术创作",
                    "DPM2": "平衡速度和质量",
                    "DPM2 a": "改进版DPM2",
                    "DPM++ 2S a": "单步优化版本",
                    "DDIM": "确定性采样，可重现结果"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"获取采样器列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取采样器列表失败: {str(e)}") 