# FPM 集成说明文档

这个文档描述了如何在 lm-evaluation-harness 中使用 FPM (Function-specific Pretrained Modules) 补丁。

## 概述

FPM 集成允许你在 MoE (Mixture of Experts) 模型中指定使用特定的专家子集，并监控专家使用情况。

## 主要改进

### 1. vLLM 支持

- **参数**: `path_to_fpm_vllm_patch`, `use_experts`
- **功能**: 动态配置转换、专家路由控制、损失监控
- **兼容性**: 确保不使用补丁时行为完全不变

### 2. 专家 Mask 创建

- **函数**: `_create_experts_mask_from_parsed_str`
- **功能**: 从配置字符串创建专家掩码
- **错误处理**: 处理无效配置和边界情况

### 3. 参数传递

- **命令行**: `--use_experts`, `--statistics_moe_experts`
- **模型参数**: 通过模型配置传递
- **任务级别**: 在任务构建时处理

## 使用方法

### 1. 基本使用 (不使用 FPM)

```bash
lm_eval --model vllm \
    --model_args "pretrained=/path/to/model,dtype=bfloat16" \
    --tasks fpm_wiki_culture \
    --batch_size 32
```

### 2. 使用 FPM 补丁

```bash
lm_eval --model vllm \
    --model_args "pretrained=/path/to/model,dtype=bfloat16,path_to_fpm_vllm_patch=./fpm_vllm_patch.py,use_experts=common_2" \
    --tasks fpm_wiki_culture \
    --batch_size 32 \
    --use_experts common_2
```

### 3. 启用专家统计

```bash
lm_eval --model vllm \
    --model_args "pretrained=/path/to/model,dtype=bfloat16,path_to_fpm_vllm_patch=./fpm_vllm_patch.py" \
    --tasks fpm_wiki_culture \
    --batch_size 32 \
    --statistics_moe_experts
```

## 参数说明

### 模型参数

- `path_to_fpm_vllm_patch`: FPM 补丁文件路径
- `use_experts`: 专家使用配置 (如 "common_2", "math_1")

### 命令行参数

- `--use_experts`: 指定要使用的专家
- `--statistics_moe_experts`: 启用专家使用统计

## 专家配置格式

### 当前支持的格式

- `common_K`: 使用前 K 个通用专家
- 其他格式可根据需要扩展

### 示例

```python
# 使用前 2 个专家
use_experts = "common_2"

# 使用特定专家 (未来扩展)
use_experts = "math_1,code_0"
```

## 验证测试

运行验证脚本确保集成正确：

```bash
# 基本验证
python validate_fpm_integration.py

# 详细输出
python validate_fpm_integration.py --verbose

# 完整集成测试
python test_fpm_integration.py
```

## 错误处理

### 常见问题

1. **FPM 补丁加载失败**
   - 检查补丁文件路径是否正确
   - 确保文件权限可读

2. **专家数量不匹配**
   - 检查模型配置中的专家数量
   - 确保 use_experts 配置合理

3. **设备不匹配**
   - 补丁会自动处理设备转换
   - 如有问题检查 CUDA/NPU 可用性

### 错误日志

FPM 集成会提供详细的日志信息：

```
[FPM Patch] Loading FPM vLLM patch from ./fpm_vllm_patch.py
[FPM Patch] ✅ FPM vLLM patch applied and model path registered
[FPM Patch] Initializing expert masks for 4 experts
[FPM Patch] Enabled first 2 experts for 'common' policy
```

## 代码架构

### 关键组件

1. **FPM 补丁 (`fpm_vllm_patch.py`)**
   - 核心补丁逻辑
   - 模型配置转换
   - 专家路由功能

2. **vLLM 集成 (`vllm_causallms.py`)**
   - 补丁加载
   - 专家 mask 初始化
   - 推理时路由控制

3. **评估器集成 (`evaluator.py`)**
   - 专家 mask 创建
   - 统计功能
   - 参数处理

### 数据流

```
命令行参数 → 模型配置 → FPM 补丁 → 专家路由 → 推理结果
     ↓
   任务构建 → 请求处理 → 批处理 → 专家选择 → 输出生成
```

## 性能考虑

### 内存使用

- 专家 mask 占用少量额外内存
- 金标准索引为可选功能
- 损失计算开销最小

### 推理速度

- FPM 路由开销很小
- 专家子集使用可能提高效率
- 统计收集有轻微开销

## 故障排除

### 调试步骤

1. **检查导入**
   ```python
   from lm_eval.models.vllm_causallms import VLLM
   print("导入成功")
   ```

2. **验证补丁**
   ```python
   # 检查补丁文件存在
   import os
   assert os.path.exists("./fpm_vllm_patch.py")
   ```

3. **测试专家 mask**
   ```python
   from lm_eval.evaluator import _create_experts_mask_from_parsed_str
   mask = _create_experts_mask_from_parsed_str({"common": -1}, {"common": [0, 1, 2, 3]})
   print(mask)
   ```

### 兼容性检查

- vLLM 版本 >= 0.7.x
- PyTorch 版本兼容
- NPU/CUDA 驱动正常

## 更新日志

### 当前版本改进

- 动态专家数量检测
- 改进的错误处理
- 更好的日志记录
- 完整的向后兼容性
- 增强的验证测试

### 未来计划

- 支持更多专家选择策略
- 动态专家权重调整
- 实时统计监控
- 自动化性能优化

## 贡献指南

如需添加新功能或修复问题：

1. 确保所有测试通过
2. 添加相应的验证测试
3. 更新文档
4. 保持向后兼容性

## 联系方式

如有问题或建议，请通过相关渠道联系开发团队。 