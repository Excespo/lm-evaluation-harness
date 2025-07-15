import os, sys
from importlib.util import spec_from_file_location, module_from_spec
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2MoeForCausalLM, PretrainedConfig
from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM as VLLMQwen2MoeForCausalLM


def load_hf_moe_patch():
    
    def _load(script, spec_id):
        spec = spec_from_file_location(str(spec_id), script)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    model_patch = _load("/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/moe_monkey_patch.py", 0)
    cfg_patch = _load("/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/src/model/config_patch.py", 1)
    
    return model_patch, cfg_patch


dummy_ckpt_dir = os.path.expanduser("~/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS/final")

# cfg for train-time config
# dummy_cfg = AutoConfig.from_pretrained(dummy_ckpt_dir)
# print(f"{dummy_cfg=}")

# patch cfg to infer-time
model_patch, cfg_patch = load_hf_moe_patch()
FPMConfig = cfg_patch.FPMConfig # getattr(cfg_patch, "FPMConfig", None)

def patch_vllm_config():

    class VLLMFPMConfig(FPMConfig):
        model_type = "qwen2_moe"
        architectures = ["Qwen2MoeForCausalLM"]
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.decoder_sparse_step = 1
            self.moe_intermediate_size = self.intermediate_size * self.n_shared_experts
            self.num_experts_per_tok = self.n_top_router
            self.num_experts = self.n_routed_experts

    FPMConfig.__name__ = "Qwen2MoeConfig"
    transformers.models.qwen2.configuration_qwen2.Qwen2MoeConfig = VLLMFPMConfig
            

model_patch.apply()
# hf_patched_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=dummy_ckpt_dir)
# hf_patched_state_dict = hf_patched_qwen2.state_dict()
# print(f"{hf_patched_qwen2=}")

patch_vllm_config()
vllm_patched_cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path=dummy_ckpt_dir)
print(f"{vllm_patched_cfg}")
vllm_patched_model = AutoModelForCausalLM.from_config(config=vllm_patched_cfg)
print(f"{vllm_patched_model=}")