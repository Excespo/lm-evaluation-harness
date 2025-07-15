from fpm_vllm_config_patch import apply as apply_cfg
from fpm_vllm_modeling_patch import apply as apply_vllm

import torch, torch_npu
from vllm import LLM
from vllm.attention import AttentionMetadata
from vllm_ascend.attention import AscendMetadata

import os, sys
sys.path.append(os.path.expanduser("~/src/FPM/src/model"))
from moe_monkey_patch import apply as apply_hf

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


apply_cfg()
apply_vllm()

ckpt = "/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/outputs/ckpts/init/Qwen2.5-MoE-6.8B-en-wiki-init-0+1-clone-culture"


def compute_hidden_states(model, input_ids):
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
    
    # kv caches
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_kv_heads = model.config.num_key_value_heads
    head_size = hidden_size // model.config.num_attention_heads
    
    kv_caches = []
    for _ in range(num_layers):
        kv_cache = torch.zeros(
            (1, 16, num_kv_heads, head_size),
            dtype=torch.bfloat16,
            device=input_ids.device
        )
        kv_caches.append(kv_cache)

    attn_metadata = AscendMetadata(
        # 从AttentionMetadata继承的必需参数
        num_prefills=1,  # 1个prefill请求
        num_prefill_tokens=seq_len,  # prefill的token数量
        num_decode_tokens=0,  # 没有decode请求
        slot_mapping=torch.arange(seq_len, dtype=torch.long),  # slot映射
        multi_modal_placeholder_index_maps=None,  # 没有多模态
        enable_kv_scales_calculation=True,  # 启用KV scales计算
        
        # AscendMetadata特有的必需参数
        max_prefill_seq_len=seq_len,  # 最大prefill序列长度
        max_decode_seq_len=0,  # 没有decode请求
        chunked_prefill_enabled=False,  # 禁用chunked prefill
        block_tables=None,  # 没有block tables
        seq_lens_tensor=torch.tensor([seq_len], dtype=torch.long),  # 序列长度张量
        
        # 可选参数（有默认值）
        seq_lens=[seq_len],  # 序列长度列表
        query_lens=[seq_len],  # 查询长度列表
        max_query_len=seq_len,  # 最大查询长度
        context_lens_tensor=torch.tensor([seq_len], dtype=torch.long),  # 上下文长度张量
    )
    
    return model(
        input_ids=input_ids,
        positions=positions,
        kv_caches=kv_caches,
        attn_metadata=attn_metadata,
        intermediate_tensors=None,
        inputs_embeds=None,
    ) # hidden_states


llm = LLM(
    model=ckpt,
    dtype="bfloat16",
    max_model_len=4096,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    trust_remote_code=True
)
vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
vllm_state_dict = vllm_model.state_dict()

device = next(llm.llm_engine.model_executor.driver_worker.model_runner.model.parameters()).device
input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
positions = torch.arange(input_ids.shape[1], device=device)
attention_mask = torch.ones_like(input_ids, device=device)

vllm_outputs = compute_hidden_states(vllm_model, input_ids=input_ids)
print(f"{vllm_outputs}")

apply_hf(set_deepspeed_parallelism=False)

hf_model = AutoModelForCausalLM.from_pretrained(ckpt, device_map={"": "npu:0"})
hf_state_dict = hf_model.state_dict()

with torch.no_grad():  
    hf_outputs = hf_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)  
    hf_hidden_states = hf_outputs.hidden_states[-1] 
print(f"{hf_outputs=}")

# Check weights
# def check(s1, s2, name):
#     if "mlp" not in name:
#         t2_name = name
#     else:
#         pref, suff = name.split(".mlp.")
#         t2_name = f"{pref}.routed_experts.deepspeed_moe.experts.deepspeed_experts.{EXPERT_ID}.{suff}"
#     print(f"Checking {name} from s1 dense, and {t2_name} from s2 moe")

#     all_close = torch.allclose(s1[name], s2[t2_name])
#     if all_close:
#         print("All close !")
#         return True
#     else:
#         print("Not close .")
#         return False


# import collections
# results = collections.defaultdict(list)
# for name in dense.keys():
#     if not name.startswith("model.layers."):
#         results["outside_decoder"].append(check(dense, moe, name))
#     else:
#         layer_id = name.split(".")[2]
#         results[layer_id].append(check(dense, moe, name))
# print(results)