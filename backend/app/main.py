"""
主应用程序入口
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from contextlib import asynccontextmanager

from .api import ai_generation
from .core.dependencies import initialize_services, shutdown_services

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    try:
        logger.info("正在启动应用...")
        await initialize_services()
        logger.info("应用启动完成")
        yield
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        raise
    finally:
        # 关闭时清理资源
        try:
            logger.info("正在关闭应用...")
            await shutdown_services()
            logger.info("应用关闭完成")
        except Exception as e:
            logger.error(f"应用关闭失败: {str(e)}")

# 创建FastAPI应用
app = FastAPI(
    title="军事目标数据集生成平台",
    description="基于传统图像合成和AI生成的军事目标数据集生成与管理平台",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(ai_generation.router, prefix="/api")

# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "军事目标数据集生成平台 API",
        "version": "1.0.0",
        "status": "running"
    }

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "message": "服务运行正常"
    }

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
