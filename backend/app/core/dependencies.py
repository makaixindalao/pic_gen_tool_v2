"""
依赖注入配置
提供服务实例的单例模式管理
"""

from functools import lru_cache
from typing import Optional
import logging

from ..services.ai_generation.ai_service import AIGenerationService
from ..services.traditional_generation.traditional_service import TraditionalGenerationService

logger = logging.getLogger(__name__)

# 全局服务实例
_ai_service: Optional[AIGenerationService] = None
_traditional_service: Optional[TraditionalGenerationService] = None

def get_ai_service() -> AIGenerationService:
    """
    获取AI生成服务实例（单例）
    
    Returns:
        AIGenerationService: AI生成服务实例
    """
    global _ai_service
    
    if _ai_service is None:
        try:
            logger.info("创建AI生成服务实例")
            _ai_service = AIGenerationService()
        except Exception as e:
            logger.error(f"AI生成服务创建失败: {str(e)}")
            logger.info("AI生成服务将在首次使用时重新尝试初始化")
            # 创建一个未初始化的实例，稍后再尝试初始化
            _ai_service = AIGenerationService()
    
    return _ai_service

def get_traditional_service() -> TraditionalGenerationService:
    """
    获取传统生成服务实例（单例）
    
    Returns:
        TraditionalGenerationService: 传统生成服务实例
    """
    global _traditional_service
    
    if _traditional_service is None:
        logger.info("创建传统生成服务实例")
        _traditional_service = TraditionalGenerationService()
    
    return _traditional_service

def cleanup_services():
    """清理所有服务实例"""
    global _ai_service, _traditional_service
    
    if _ai_service:
        _ai_service.cleanup()
        _ai_service = None
        logger.info("AI生成服务已清理")
    
    if _traditional_service:
        _traditional_service.cleanup()
        _traditional_service = None
        logger.info("传统生成服务已清理")

# 应用启动时的初始化函数
async def initialize_services():
    """初始化所有服务"""
    try:
        logger.info("开始初始化服务...")
        
        # 初始化传统生成服务（总是可用）
        traditional_service = get_traditional_service()
        await traditional_service.initialize()
        logger.info("传统生成服务初始化完成")
        
        # 尝试初始化AI生成服务
        try:
            ai_service = get_ai_service()
            success = await ai_service.initialize()
            if success:
                logger.info("AI生成服务初始化完成")
            else:
                logger.warning("AI生成服务初始化失败，但服务实例已创建")
                logger.info("可通过API手动重新初始化")
        except Exception as e:
            logger.warning(f"AI生成服务初始化异常: {str(e)}")
            logger.info("AI生成服务将在模型可用时自动重试")
        
        logger.info("服务初始化流程完成")
        
    except Exception as e:
        logger.error(f"关键服务初始化失败: {str(e)}")
        # 不抛出异常，允许应用继续运行
        logger.info("应用将以降级模式运行")

# 应用关闭时的清理函数
async def shutdown_services():
    """关闭所有服务"""
    try:
        logger.info("开始关闭服务...")
        cleanup_services()
        logger.info("所有服务已关闭")
        
    except Exception as e:
        logger.error(f"服务关闭失败: {str(e)}")
        raise 