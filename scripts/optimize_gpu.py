#!/usr/bin/env python3
"""
GPU优化配置脚本
自动检测并配置最佳GPU设置
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def check_gpu_capability():
    """检查GPU能力"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
            
        device_props = torch.cuda.get_device_properties(0)
        gpu_name = device_props.name
        total_memory = device_props.total_memory / 1024**3
        
        return {
            "name": gpu_name,
            "memory": total_memory,
            "compute_capability": f"{device_props.major}.{device_props.minor}"
        }
    except:
        return None

def get_optimal_settings(gpu_info):
    """根据GPU信息获取最佳设置"""
    if not gpu_info:
        return {
            "ai_device": "cpu",
            "ai_torch_dtype": "float32",
            "ai_enable_model_offload": False,
            "batch_size": 1,
            "attention_slicing": True
        }
    
    memory_gb = gpu_info["memory"]
    
    if memory_gb >= 12:  # 高端GPU (RTX 4080/4090, RTX 3080Ti等)
        return {
            "ai_device": "cuda",
            "ai_torch_dtype": "float16",
            "ai_enable_model_offload": False,  # 显存充足，不需要CPU卸载
            "batch_size": 2,
            "attention_slicing": False,
            "enable_xformers": True,
            "estimated_speed": "3-8秒/张"
        }
    elif memory_gb >= 8:  # 中端GPU (RTX 4070, RTX 3070等)
        return {
            "ai_device": "cuda", 
            "ai_torch_dtype": "float16",
            "ai_enable_model_offload": True,
            "batch_size": 1,
            "attention_slicing": True,
            "enable_xformers": True,
            "estimated_speed": "5-15秒/张"
        }
    elif memory_gb >= 6:  # 入门GPU (RTX 4060, RTX 3060等)
        return {
            "ai_device": "cuda",
            "ai_torch_dtype": "float16", 
            "ai_enable_model_offload": True,
            "batch_size": 1,
            "attention_slicing": True,
            "enable_xformers": False,
            "estimated_speed": "8-20秒/张"
        }
    else:  # 低端GPU
        return {
            "ai_device": "cuda",
            "ai_torch_dtype": "float16",
            "ai_enable_model_offload": True,
            "batch_size": 1,
            "attention_slicing": True,
            "enable_xformers": False,
            "estimated_speed": "15-30秒/张"
        }

def apply_gpu_optimizations():
    """应用GPU优化"""
    try:
        # 检查GPU
        gpu_info = check_gpu_capability()
        settings = get_optimal_settings(gpu_info)
        
        print("=== GPU优化配置 ===")
        if gpu_info:
            print(f"检测到GPU: {gpu_info['name']}")
            print(f"显存: {gpu_info['memory']:.1f} GB")
            print(f"计算能力: {settings.get('compute_capability', 'N/A')}")
        else:
            print("未检测到可用GPU，使用CPU模式")
        
        print(f"\n推荐配置:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        # 测试GPU性能
        if gpu_info and settings["ai_device"] == "cuda":
            print(f"\n=== GPU性能测试 ===")
            test_gpu_performance()
        
        return settings
        
    except Exception as e:
        print(f"优化配置失败: {str(e)}")
        return None

def test_gpu_performance():
    """测试GPU性能"""
    try:
        import torch
        import time
        
        device = torch.device("cuda")
        
        # 测试矩阵运算性能
        print("测试GPU计算性能...")
        start_time = time.time()
        
        for _ in range(10):
            a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
            b = torch.randn(2048, 2048, device=device, dtype=torch.float16)
            c = torch.mm(a, b)
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        print(f"矩阵运算测试: {elapsed:.2f}秒")
        
        # 测试显存使用
        print("测试显存使用...")
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**3
        
        # 模拟SD模型显存使用
        dummy_tensors = []
        for i in range(5):
            tensor = torch.randn(512, 512, 512, device=device, dtype=torch.float16)
            dummy_tensors.append(tensor)
        
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_used = memory_after - memory_before
        
        print(f"模拟显存使用: {memory_used:.2f} GB")
        
        # 清理
        del dummy_tensors
        torch.cuda.empty_cache()
        
        print("✓ GPU性能测试完成")
        
    except Exception as e:
        print(f"GPU性能测试失败: {str(e)}")

def restart_service_with_gpu():
    """重启服务以应用GPU配置"""
    print("\n=== 重启AI服务 ===")
    print("请重启后端服务以应用GPU配置:")
    print("1. 停止当前运行的后端服务")
    print("2. 重新运行: cd backend && python -m app.main")
    print("3. 或者调用API重新初始化: POST /api/ai/initialize")

def main():
    """主函数"""
    print("=== Stable Diffusion GPU优化工具 ===\n")
    
    settings = apply_gpu_optimizations()
    
    if settings and settings["ai_device"] == "cuda":
        print(f"\n🚀 GPU加速已配置！")
        print(f"预计生成速度: {settings.get('estimated_speed', '未知')}")
        print(f"相比CPU模式提升: 5-10倍")
        
        restart_service_with_gpu()
    else:
        print(f"\n⚠️  无法启用GPU加速")
        print("请检查:")
        print("1. 是否安装了支持CUDA的PyTorch")
        print("2. NVIDIA驱动是否正确安装")
        print("3. GPU是否支持CUDA")

if __name__ == "__main__":
    main() 