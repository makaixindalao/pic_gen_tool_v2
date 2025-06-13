"""
前端AI生成服务客户端
与后端AI生成API进行通信
"""

import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AIServiceClient:
    """AI生成服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化AI服务客户端
        
        Args:
            base_url: 后端API基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/ai"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info(f"AI服务客户端初始化，API地址: {self.api_base}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            **kwargs: 请求参数
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        url = f"{self.api_base}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {method} {url}, 错误: {str(e)}")
            raise Exception(f"API请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"响应解析失败: {str(e)}")
            raise Exception(f"响应解析失败: {str(e)}")
    
    def initialize_service(self) -> bool:
        """
        初始化AI生成服务
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            response = self._make_request('POST', '/initialize')
            return response.get('success', False)
            
        except Exception as e:
            logger.error(f"服务初始化失败: {str(e)}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        获取服务状态
        
        Returns:
            Dict[str, Any]: 服务状态信息
        """
        try:
            response = self._make_request('GET', '/status')
            return response.get('data', {})
            
        except Exception as e:
            logger.error(f"获取服务状态失败: {str(e)}")
            return {}
    
    def build_prompt(
        self,
        military_target: str,
        weather: str,
        scene: str,
        custom_prompt: str = "",
        style_strength: float = 0.7,
        technical_detail: float = 0.8
    ) -> Tuple[str, str]:
        """
        构建提示词
        
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
        try:
            data = {
                "military_target": military_target,
                "weather": weather,
                "scene": scene,
                "custom_prompt": custom_prompt,
                "style_strength": style_strength,
                "technical_detail": technical_detail
            }
            
            response = self._make_request('POST', '/build-prompt', json=data)
            
            if response.get('success'):
                prompt_data = response.get('data', {})
                return (
                    prompt_data.get('positive_prompt', ''),
                    prompt_data.get('negative_prompt', '')
                )
            else:
                return "", ""
                
        except Exception as e:
            logger.error(f"构建提示词失败: {str(e)}")
            return "", ""
    
    def generate_images(
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
        generate_annotations: bool = True
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
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        try:
            data = {
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
                "technical_detail": technical_detail,
                "save_images": save_images,
                "generate_annotations": generate_annotations
            }
            
            response = self._make_request('POST', '/generate', json=data)
            
            if response.get('success'):
                return response.get('data', {})
            else:
                raise Exception(response.get('message', '生成失败'))
                
        except Exception as e:
            logger.error(f"图像生成失败: {str(e)}")
            raise
    
    def get_prompt_suggestions(self, military_target: str) -> Dict[str, List[str]]:
        """
        获取提示词建议
        
        Args:
            military_target: 军事目标类型
            
        Returns:
            Dict[str, List[str]]: 提示词建议
        """
        try:
            response = self._make_request('GET', f'/prompt-suggestions/{military_target}')
            
            if response.get('success'):
                return response.get('data', {})
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取提示词建议失败: {str(e)}")
            return {}
    
    def optimize_prompt(self, prompt: str) -> str:
        """
        优化提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 优化后的提示词
        """
        try:
            response = self._make_request('POST', '/optimize-prompt', params={'prompt': prompt})
            
            if response.get('success'):
                data = response.get('data', {})
                return data.get('optimized_prompt', prompt)
            else:
                return prompt
                
        except Exception as e:
            logger.error(f"提示词优化失败: {str(e)}")
            return prompt
    
    def get_generation_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        获取生成预设配置
        
        Returns:
            Dict[str, Dict[str, Any]]: 预设配置
        """
        try:
            response = self._make_request('GET', '/presets')
            
            if response.get('success'):
                return response.get('data', {})
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取预设配置失败: {str(e)}")
            return {}
    
    def get_available_schedulers(self) -> Dict[str, Any]:
        """
        获取可用的采样器列表
        
        Returns:
            Dict[str, Any]: 采样器信息
        """
        try:
            response = self._make_request('GET', '/schedulers')
            
            if response.get('success'):
                return response.get('data', {})
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取采样器列表失败: {str(e)}")
            return {}
    
    def get_generation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取生成历史
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            List[Dict[str, Any]]: 生成历史记录
        """
        try:
            response = self._make_request('GET', '/generation-history', params={'limit': limit})
            
            if response.get('success'):
                data = response.get('data', {})
                return data.get('history', [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取生成历史失败: {str(e)}")
            return []
    
    def clear_generation_history(self) -> bool:
        """
        清空生成历史
        
        Returns:
            bool: 操作是否成功
        """
        try:
            response = self._make_request('DELETE', '/generation-history')
            return response.get('success', False)
            
        except Exception as e:
            logger.error(f"清空生成历史失败: {str(e)}")
            return False
    
    def download_image(self, generation_id: str, image_index: int, save_path: str) -> bool:
        """
        下载生成的图像
        
        Args:
            generation_id: 生成ID
            image_index: 图像索引
            save_path: 保存路径
            
        Returns:
            bool: 下载是否成功
        """
        try:
            url = f"{self.api_base}/download/{generation_id}/{image_index}"
            
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"图像下载成功: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"图像下载失败: {str(e)}")
            return False
    
    def batch_generate(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量生成图像
        
        Args:
            configs: 生成配置列表
            
        Returns:
            Dict[str, Any]: 批量生成结果
        """
        try:
            data = {"configs": configs}
            
            response = self._make_request('POST', '/batch-generate', json=data)
            
            if response.get('success'):
                return response.get('data', {})
            else:
                raise Exception(response.get('message', '批量生成失败'))
                
        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        try:
            response = self._make_request('GET', '/model-info')
            
            if response.get('success'):
                return response.get('data', {})
            else:
                return {}
                
        except Exception as e:
            logger.error(f"获取模型信息失败: {str(e)}")
            return {}
    
    def close(self):
        """关闭客户端连接"""
        if self.session:
            self.session.close()
            logger.info("AI服务客户端连接已关闭") 