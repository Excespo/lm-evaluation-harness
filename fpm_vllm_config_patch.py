import os
import json

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2Config, Qwen2Tokenizer, Qwen2ForCausalLM
from transformers import Qwen2MoeConfig

import vllm.model_executor.models.qwen2
import vllm.model_executor.models.qwen2_moe

def apply():
    

    class VLLMFPMConfig(Qwen2MoeConfig):
        model_type = "qwen2"
        
        def __init__(self, *args, n_shared_experts=2, n_routed_experts=2, n_top_router=1, **kwargs):
            super().__init__(**kwargs)
            self.n_shared_experts = n_shared_experts
            self.n_routed_experts = n_routed_experts
            self.n_top_router = n_top_router

            # self.decoder_sparse_step = 1
            # self.norm_topk_prob = False
            # self.output_router_logits = False

            self.moe_intermediate_size = self.intermediate_size
            self.shared_expert_intermediate_size = self.intermediate_size * self.n_shared_experts
            self.num_experts_per_tok = self.n_top_router
            self.num_experts = self.n_routed_experts

        @property
        def function_to_expert_indices(self):
            self.index_file_path = os.path.join(self._name_or_path, "function_to_expert.index.json")
            if not os.path.exists(self.index_file_path):
                return None
            return json.loads(open(self.index_file_path, "rt", encoding="utf-8").read())

    VLLMFPMConfig.__name__ = "Qwen2Config"
    transformers.models.qwen2.configuration_qwen2.Qwen2Config = VLLMFPMConfig
    
    original_qwen2_config = transformers.models.qwen2.configuration_qwen2.Qwen2Config
    VLLMFPMConfig.__name__ = "Qwen2Config"
    transformers.models.qwen2.configuration_qwen2.Qwen2Config = VLLMFPMConfig
    AutoConfig.register("qwen2", VLLMFPMConfig, exist_ok=True)
    AutoModelForCausalLM.register(VLLMFPMConfig, Qwen2ForCausalLM, exist_ok=True)
    AutoTokenizer.register(VLLMFPMConfig, Qwen2Tokenizer, exist_ok=True)


if __name__ == "__main__":
    pretrained = "/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS/final"
    cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained)
    print(cfg) # yes