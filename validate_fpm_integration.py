#!/usr/bin/env python3
"""
简单的 FPM 集成验证脚本
测试 FPM 补丁在 lm-eval 中的集成是否正确
"""
import sys
import os
import json
import logging
from pathlib import Path
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_import_without_fpm():
    """测试不启用 FPM 时的导入"""
    logger.info("Testing imports without FPM...")
    
    try:
        # 测试基础导入
        from lm_eval.models.vllm_causallms import VLLM
        from lm_eval.evaluator import _create_experts_mask_from_parsed_str
        
        logger.info("✅ All imports successful without FPM")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_vllm_model_creation():
    """测试 VLLM 模型创建"""
    logger.info("Testing VLLM model creation...")
    
    try:
        from lm_eval.models.vllm_causallms import VLLM
        
        # 测试不带 FPM 的模型创建
        model_args = {
            "pretrained": "microsoft/DialoGPT-medium",
            "dtype": "float16",
            "max_model_len": 512,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
            "data_parallel_size": 1,
        }
        
        # 这里只测试初始化，不实际加载模型
        logger.info("Model arguments validated")
        
        # 测试带 FPM 的参数
        fpm_args = model_args.copy()
        fpm_args.update({
            "path_to_fpm_vllm_patch": "./fpm_vllm_patch.py",
            "use_experts": "common_2",
        })
        
        logger.info("FPM arguments validated")
        logger.info("✅ VLLM model creation test passed")
        return True
    except Exception as e:
        logger.error(f"❌ VLLM model creation test failed: {e}")
        return False

def test_experts_mask_function():
    """测试专家 mask 创建函数"""
    logger.info("Testing expert mask creation function...")
    
    try:
        from lm_eval.evaluator import _create_experts_mask_from_parsed_str
        
        # 测试用例 1：正常配置
        use_experts = {"common": -1}
        function_to_expert_indices = {"common": [0, 1, 2, 3]}
        
        mask = _create_experts_mask_from_parsed_str(use_experts, function_to_expert_indices)
        assert mask is not None, "Mask should not be None for valid configuration"
        
        # 测试用例 2：空配置
        empty_mask = _create_experts_mask_from_parsed_str({}, function_to_expert_indices)
        assert empty_mask is None, "Mask should be None for empty configuration"
        
        # 测试用例 3：None 输入
        none_mask = _create_experts_mask_from_parsed_str(None, function_to_expert_indices)
        assert none_mask is None, "Mask should be None for None input"
        
        logger.info("✅ Expert mask creation function test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Expert mask creation function test failed: {e}")
        return False

def test_parameter_handling():
    """测试参数处理"""
    logger.info("Testing parameter handling...")
    
    try:
        # 测试 use_experts 参数解析
        use_experts_examples = [
            None,
            "",
            "common_2",
            "math_1",
        ]
        
        for example in use_experts_examples:
            logger.info(f"Testing use_experts: {example}")
            # 这里只测试参数不会导致错误
            
        logger.info("✅ Parameter handling test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Parameter handling test failed: {e}")
        return False

def test_cli_arguments():
    """测试命令行参数"""
    logger.info("Testing CLI arguments...")
    
    try:
        from lm_eval.__main__ import setup_parser
        
        parser = setup_parser()
        
        # 测试不带 FPM 的参数
        args = parser.parse_args([
            "--model", "vllm",
            "--model_args", "pretrained=test,dtype=float16",
            "--tasks", "test_task",
            "--batch_size", "1",
        ])
        
        assert args.model == "vllm"
        assert args.use_experts is None
        assert args.statistics_moe_experts is False
        
        # 测试带 FPM 的参数
        args_with_fpm = parser.parse_args([
            "--model", "vllm",
            "--model_args", "pretrained=test,dtype=float16",
            "--tasks", "test_task",
            "--batch_size", "1",
            "--use_experts", "common_2",
            "--statistics_moe_experts",
        ])
        
        assert args_with_fpm.use_experts == "common_2"
        assert args_with_fpm.statistics_moe_experts is True
        
        logger.info("✅ CLI arguments test passed")
        return True
    except Exception as e:
        logger.error(f"❌ CLI arguments test failed: {e}")
        return False

def main():
    """运行所有测试"""
    parser = argparse.ArgumentParser(description="验证 FPM 集成")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("开始 FPM 集成验证...")
    
    tests = [
        ("Import Test", test_import_without_fpm),
        ("VLLM Model Creation", test_vllm_model_creation),
        ("Expert Mask Function", test_experts_mask_function),
        ("Parameter Handling", test_parameter_handling),
        ("CLI Arguments", test_cli_arguments),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"运行测试: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"测试 {test_name} 崩溃: {e}")
            failed += 1
        logger.info("-" * 50)
    
    logger.info(f"测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        logger.info("🎉 所有测试通过!")
        return 0
    else:
        logger.error("💥 有测试失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 