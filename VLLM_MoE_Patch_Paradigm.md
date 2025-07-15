### VLLM MoE Patch 范式解析

#### 1. 核心目标与补丁 (`fpm_vllm_patch.py`) 简介

本范式的核心目标是在 `lm-evaluation-harness` 框架下，为基于 `vllm` 的 FPM（Fine-grained Post-training Module）模型动态添加**专家路由控制**（Expert Masking）功能。

`fpm_vllm_patch.py` 脚本通过 Monkey Patch 的方式，在不修改原始库代码的前提下，重载了 `vllm` 的配置加载、权重转换和 MoE 路由函数，使它们能够接收并处理 `experts_mask` 参数，实现专家子集选择功能。

**关键机制**:
- **Expert Mask**: 使用 `0.0`（启用）和 `-inf`（禁用）控制专家选择
- **全局上下文**: 通过 `FpmRoutingContext` 传递路由参数，避免修改函数签名
- **自动转换**: 将 FPM DeepSpeed 格式自动转换为 vLLM 兼容格式

#### 2. `lm-evaluation-harness` (VLLM侧) 如何与补丁协作

VLLM 范式延续了 HF 范式的**"框架配合，补丁实现"**核心思想，但采用了更加自动化的方法。整个流程由测试脚本驱动。

**关键协作点:**

1. **触发补丁执行**:
   - 测试脚本中最关键的参数是：
     ```bash
     model_args="...path_to_fmp_vllm_patch=$fpm_patch_path,use_experts=culture"
     ```
   - 当 `lm_eval` 以 `--model vllm` 模式启动时，`vllm_causallms.py` 检测到 `path_to_fmp_vllm_patch` 参数后，会在模型实例化*之前*动态加载并执行补丁脚本。
   - 补丁脚本的 `apply_fpm_vllm_patch()` 函数会系统性地替换 vLLM 内部的关键组件，包括配置加载器、权重加载器和路由函数。

2. **传递专家选择参数**:
   - 我们希望能够从命令行控制使用哪些专家，测试脚本通过 `use_experts=<value>` 实现这一点。
   - `lm_eval` 将 `use_experts` 参数传递给 `vllm_causallms.py`，后者根据此参数生成 `experts_mask` 张量。
   - **重要**: 从字符串 (`use_experts="culture"`) 到张量 (`experts_mask`) 的转换，是通过读取模型检查点中的 `function_to_expert.index.json` 映射文件完成的：
     ```json
     {
         "culture": [0, 1, 2],
         "science": [3, 4, 5],
         "common": [0, 1, 2, 3, 4, 5]
     }
     ```

3. **全局上下文传递**:
   - 不同于 HF 范式通过函数参数层层传递 `experts_mask`，VLLM 范式使用全局上下文管理器 `FpmRoutingContext`。
   - 在推理入口点，通过上下文管理器设置专家掩码：
     ```python
     with fpm_routing_context(experts_mask=self.experts_mask):
         outputs = self._model_generate(...)
     ```
   - 在路由计算点，从全局上下文获取掩码并应用。

#### 3. 调用链条

1. **[Shell]** 测试脚本执行 `lm_eval` 命令，传入模型路径、任务名，以及两个关键参数：`path_to_fmp_vllm_patch` 和 `use_experts`。

2. **[lm-eval-harness]** `lm_eval` 框架启动：
   a. `vllm_causallms.py` 解析 `model_args`，发现 `path_to_fmp_vllm_patch` 参数。
   b. 立即执行 `fpm_vllm_patch.py`，完成对 vLLM 核心组件的 Monkey Patch。
   c. 调用 `register_fpm_model()` 将当前模型路径注册为 FPM 模型。
   d. 根据 `use_experts` 参数和映射文件初始化专家掩码。

3. **[vLLM Model Loading]** vLLM 开始加载模型：
   a. **配置转换**: `_patched_get_config` 拦截配置加载，自动将 Qwen2 配置转换为 Qwen2MoE 配置。
   b. **权重转换**: `_patched_qwen2moeforcausallm_load_weights` 拦截权重加载，通过 `_convert_weight_names` 自动转换权重名称。
   c. **路由劫持**: `_patched_qwen2moesparsemoeblock_forward` 将自定义路由函数注入到 MoE 层。

4. **[Python-Inference]** `lm_eval` 在评估循环中调用推理：
   ```python
   # 在 vllm_causallms.py 的 generate_until 中
   with fmp_routing_context(experts_mask=self.experts_mask) as ctx:
       outputs = self._model_generate(...)
   ```

5. **[vLLM Runtime]** vLLM 执行推理：
   a. 请求经过标准的 vLLM 处理流水线。
   b. 在 MoE 层的路由计算中，`fpm_custom_routing_function` 被调用。
   c. 函数从全局上下文获取 `experts_mask`，修改 `gating_output`：
      ```python
      gating_output = gating_output + experts_mask  # 应用专家掩码
      ```
   d. 后续的 top-k 选择和专家调度按修改后的 logits 进行。

6. **[Result]** 推理结果返回给 `lm-eval` 框架进行评估。

**小结**: VLLM 范式是 HF 范式在推理加速框架下的进化版本。它保持了**"框架配合，补丁实现"**的核心思想，但通过**全局上下文**和**自动转换**机制，实现了更高的易用性和更低的侵入性。这种设计特别适合在生产环境中部署 FPM 模型，提供了开箱即用的用户体验。 