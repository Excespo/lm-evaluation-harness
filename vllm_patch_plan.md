1. 在 `vllm_causallm.py` 中加载补丁，补丁的形式和huggingface一样，注意vllm逻辑下的dp多进程奇怪的写法
2. 在 `vllm_causallm.py` 中修改dp每个进程的资源管理，避免全部在同一张卡上抢占资源
3. 验证 config 和 补丁 （正确计算即可）的功能
4. expert mask和use experts, 后者作为lm_eval主入口的新参数，在 `__main__.py` 中统一加入，通过 `evaluator.py` 传入
    - huggingface 具体调用在 `lm_eval.api.task.Task.build_all_requests` -> `lm_eval.api.task.ConfigurableTask.construct_requests` 然后计算出expert mask (我的理想设计是把mask直接注册给model, 然后model补丁直接通过模型是否具备属性来管理router结果)
    - vllm和huggingface的区别在expert mask的实现不是通过作为forward的入参，而是作为模型本身的属性，全局统一的expert mask
    - 因此vllm的改动其实不涉及 Task，但是需要考虑到目前huggingface修改了ConfigurableTask！因此这里是有矛盾的，需要暂时性的特殊处理一下
5. stats_moe功能，同样是在主入口完成，具体的统计逻辑通过注册钩子，查看 `evaluator.py`，应该是和huggingface统一完成的 （是否需要fpm patch和moe stats同时生效？）

fix:
1. 加载参数的函数逻辑：AutoWeightsLoader，核心逻辑在_load_module, 做法是递归找到key的前缀，一直去匹配模型结构直到最底层，本质上BFS，但是我没有想明白为什么同样这个函数可以直接用来加载HF的模型，我先转到hf再直接调用就不可以？- 暂时通过版本对齐解决了！
2. 经典的expert mask没有按照-1正确view的bug