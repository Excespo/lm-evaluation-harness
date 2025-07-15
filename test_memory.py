import torch
import os
from vllm.engine.arg_utils import EngineArgs
from vllm.worker.worker import Worker
from vllm.utils import get_distributed_init_method, get_ip, get_open_port

def test_memory_calculation_with_data_parallel():
    """测试不同data_parallel_size值下的内存计算"""
    
    # 测试配置
    configs = [
        {"data_parallel_size": 1, "tensor_parallel_size": 1},
        {"data_parallel_size": 8, "tensor_parallel_size": 1},
    ]
    
    for config in configs:
        print(f"\n=== 测试data_parallel_size={config['data_parallel_size']} ===")
        
        try:
            # 创建引擎参数
            engine_args = EngineArgs(
                model="your-model-path",
                dtype="bfloat16",
                max_model_len=512,
                gpu_memory_utilization=0.4,
                data_parallel_size=config["data_parallel_size"],
                tensor_parallel_size=config["tensor_parallel_size"]
            )
            
            engine_config = engine_args.create_engine_config()
            
            # 创建worker
            distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
            worker = Worker(
                vllm_config=engine_config,
                local_rank=0,
                rank=0,
                distributed_init_method=distributed_init_method,
                is_driver_worker=True,
            )
            
            # 初始化设备并检查内存
            worker.init_device()
            
            # 检查初始内存状态
            if hasattr(torch, 'npu') and torch.npu.is_available():
                free_mem, total_mem = torch.npu.mem_get_info()
                print(f"初始NPU内存：{free_mem/1024**3:.2f}GB 可用 / {total_mem/1024**3:.2f}GB 总计")
            
            # 加载模型并检查内存使用
            worker.load_model()
            
            if hasattr(torch, 'npu') and torch.npu.is_available():
                free_mem_after, total_mem = torch.npu.mem_get_info()
                used_mem = (free_mem - free_mem_after) / 1024**3
                print(f"模型加载后内存：{free_mem_after/1024**3:.2f}GB 可用，使用了{used_mem:.2f}GB")
            
            # 测试内存分析
            available_memory = worker.determine_available_memory()
            print(f"可用KV缓存内存：{available_memory/1024**3:.2f}GB")
            
        except Exception as e:
            print(f"配置{config}出错：{e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_memory_calculation_with_data_parallel()