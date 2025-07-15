import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple, Union, List, Dict, Any, Iterable, Set, Callable
import logging

from transformers import PretrainedConfig
from vllm import LLM
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
import vllm.model_executor.models.qwen2_moe
import vllm.model_executor.models.qwen2
from vllm.model_executor.models.utils import AutoWeightsLoader, is_pp_missing_parameter, WeightsMapper, PPMissingLayer
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.logger import init_logger

logger = init_logger(__name__)
logger.setLevel(logging.DEBUG)

def apply():

    ds_to_hf_mapper = WeightsMapper(  
        orig_to_new_substr={  
            "routed_experts.deepspeed_moe.experts.deepspeed_experts": "mlp.experts",  
            "routed_experts.deepspeed_moe.gate.wg": "mlp.gate",   
            "shared_experts": "mlp.shared_expert"  
        }  
    )

    def my_load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        # print(f"\nMY LOAD\n")
        # print(f"{base_prefix=}")
        # print(f"{module=}")
        if isinstance(module, PPMissingLayer):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters "
                        "for module %s", module)
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )
        
        # print("skipping logic: module != self.module")

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        # print(f"{child_modules=}")
        # print(f"{child_params=}")

        for child_prefix, child_weights in self._groupby_prefix(weights):
            # print(f"{child_prefix=}")
            prefix = self._get_qualname(base_prefix, child_prefix)
            # print(f"after getting qual name, {prefix=}")

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(prefix,
                                             child_modules[child_prefix],
                                             child_weights)
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(prefix, child_params[child_prefix],
                                            child_weights)
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                msg = (f"There is no module or parameter named '{prefix}' "
                       f"in {type(self.module).__name__}")
                raise ValueError(msg)

    def my_load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: Optional[WeightsMapper] = None,
    ) -> set[str]:
        print("\nMY LOAD\n")
        if mapper is not None:
            weights = mapper.apply(weights)
        # filter out weights with first-prefix/substr to skip in name
        weights = ((name, weight) for name, weight in weights
                   if not self._can_skip(name))
        import itertools
        import collections
        weights, w_copy = itertools.tee(weights, 2)
        names = [name for name, _ in w_copy]
        names_outside_decoder = [name for name in names if not name.startswith("model.layers")]
        names_inside_decoder = list(set(names) - set(names_outside_decoder))
        print(f"{names_outside_decoder=}")
        # print(f"{names_inside_decoder=}")
        names_inside_decoder_div_by_layers = collections.defaultdict(list)
        for name in names_inside_decoder:
            suffix = name.split("model.layers.")[1]
            # print(f"{suffix=}")
            layer_id = int(suffix.split(".")[0])
            suffix = ".".join(suffix.split(".")[1:])
            # print(f"{layer_id=}, {suffix=}")
            names_inside_decoder_div_by_layers[layer_id].append(suffix)

        layer_0 = sorted(names_inside_decoder_div_by_layers[0])
        print(f"decoder has {len(names_inside_decoder_div_by_layers.keys())} layers")
        print(f"{layer_0=}, {len(layer_0)=}")
        for layer, names in names_inside_decoder_div_by_layers.items():
            names.sort()
            # print(f"{names=}")
            print(f"{layer=}, len equal to layer 0? {len(names)==len(layer_0)}. whole list equal to layer 0? {names==layer_0}")
        has_same_keys = all(
            names == names_inside_decoder_div_by_layers[0]
            for i, names in names_inside_decoder_div_by_layers.items()
        )
        print(f"each layer has same keys: {has_same_keys}")


        # print(f"{names=}")

        autoloaded_weights = set(self._load_module("", self.module, weights))
        print("\nAuto Loaded !\n")
        return autoloaded_weights
    
    AutoWeightsLoader.load_weights = my_load_weights
    AutoWeightsLoader._load_module = my_load_module

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

    # 1. load_weights

    def qwen2moe_forcausallm_load_weights_0_7_3(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        print("Using 0.7.3 patch")
        
        print(f"pre-transforming ds ckpt to hf")
        def _convert_ds_name_to_hf_name(ds_name):
            s = ds_name.replace(
                "routed_experts.deepspeed_moe.experts.deepspeed_experts", "mlp.experts"
            ).replace(
                "routed_experts.deepspeed_moe.gate.wg", "mlp.gate"
            ).replace(
                "shared_experts", "mlp.shared_expert"
            )
            return s
        
        # def _converted_iterator():
        #     for name, weight in weights:
        #         print(f"before: {name=}")
        #         if "experts" in name:
        #             name = _convert_ds_name_to_hf_name(name)
        #         print(f"after: {name=}")
        #         yield (name, weight)
        
        converted_weights = []
        for name, weight in weights:
            # print(f"before: {name=}")
            if "experts" in name:
                name = _convert_ds_name_to_hf_name(name)
            # print(f"after: {name=}")
            converted_weights.append((name, weight))

        import itertools
        import collections
        # weights, w_copy = itertools.tee(_converted_iterator(), 2)
        weights, w_copy = itertools.tee(converted_weights, 2)
        # print(f"{weights=}")
        # print(f"{w_copy=}")
        print(f"transformed ds ckpt to hf")
        
        # names = [name for name, _ in w_copy]
        # names_outside_decoder = [name for name in names if not name.startswith("model.layers")]
        # names_inside_decoder = list(set(names) - set(names_outside_decoder))
        # print(f"{names_outside_decoder=}")
        # # print(f"{names_inside_decoder=}")
        # names_inside_decoder_div_by_layers = collections.defaultdict(list)
        # for name in names_inside_decoder:
        #     suffix = name.split("model.layers.")[1]
        #     # print(f"{suffix=}")
        #     layer_id = int(suffix.split(".")[0])
        #     suffix = ".".join(suffix.split(".")[1:])
        #     # print(f"{layer_id=}, {suffix=}")
        #     names_inside_decoder_div_by_layers[layer_id].append(suffix)

        # layer_0 = sorted(names_inside_decoder_div_by_layers[0])
        # print(f"decoder has {len(names_inside_decoder_div_by_layers.keys())} layers")
        # print(f"{layer_0=}, {len(layer_0)=}")
        # for layer, names in names_inside_decoder_div_by_layers.items():
        #     names.sort()
        #     # print(f"{names=}")
        #     print(f"{layer=}, len equal to layer 0? {len(names)==len(layer_0)}. whole list equal to layer 0? {names==layer_0}")
        # has_same_keys = all(
        #     names == names_inside_decoder_div_by_layers[0]
        #     for i, names in names_inside_decoder_div_by_layers.items()
        # )
        # print(f"each layer has same keys: {has_same_keys}")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                # print(f"Paraming {name=}")
                param = params_dict[name]
                weight_loader = param.weight_loader
                res = weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint "
                                f"(e.g. {name}), but not found the expected "
                                f"name in the model "
                                f"(e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded.")
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        # print(f"{loaded_params=}")
        # names = list(loaded_params)
        # names_outside_decoder = [name for name in names if not name.startswith("model.layers")]
        # names_inside_decoder = list(set(names) - set(names_outside_decoder))
        # print(f"{names_outside_decoder=}")
        # # print(f"{names_inside_decoder=}")
        # names_inside_decoder_div_by_layers = collections.defaultdict(list)
        # for name in names_inside_decoder:
        #     suffix = name.split("model.layers.")[1]
        #     # print(f"{suffix=}")
        #     layer_id = int(suffix.split(".")[0])
        #     suffix = ".".join(suffix.split(".")[1:])
        #     # print(f"{layer_id=}, {suffix=}")
        #     names_inside_decoder_div_by_layers[layer_id].append(suffix)

        # layer_0 = sorted(names_inside_decoder_div_by_layers[0])
        # print(f"decoder has {len(names_inside_decoder_div_by_layers.keys())} layers")
        # print(f"{layer_0=}, {len(layer_0)=}")
        # for layer, names in names_inside_decoder_div_by_layers.items():
        #     names.sort()
        #     # print(f"{names=}")
        #     print(f"{layer=}, len equal to layer 0? {len(names)==len(layer_0)}. whole list equal to layer 0? {names==layer_0}")
        # has_same_keys = all(
        #     names == names_inside_decoder_div_by_layers[0]
        #     for i, names in names_inside_decoder_div_by_layers.items()
        # )
        return loaded_params


    vllm.model_executor.models.qwen2_moe.Qwen2MoeForCausalLM.load_weights = qwen2moe_forcausallm_load_weights_0_7_3

    # 2. patch qwen2 with no shared gate and expert-mask-driven moe block, and with other moe components
    Qwen2MoeMLP = vllm.model_executor.models.qwen2_moe.Qwen2MoeMLP

    def set_expert_mask(llm, mask):
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model 
        decoder_layers = model.model.layers
        print(f"Setting expert mask = {mask} for llm of {len(decoder_layers)} layers")
        for layer in model.model.layers:
            print(f"{layer} setting {mask}")
            layer.mlp.set_expert_mask(mask)
    
    def add_mask_to_qwen2moe_blocks(self, mask_value):
        """
        为vLLM模型中所有 Qwen2MoeSparseMoeBlock 类型的模块添加 mask 属性
        
        Args:
            mask_value: 要设置的mask值
        """
        print(f"Adding mask attribute to Qwen2MoeSparseMoeBlock modules with value: {mask_value}")
        
        # 获取所有模块的字典，类似于原函数中的 params_dict = dict(self.named_parameters())
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        modules_dict = dict(model.named_modules())
        # print(f"{modules_dict=}")
        modified_modules = []
        
        # 遍历所有模块，寻找 Qwen2MoeSparseMoeBlock 类型的模块
        for name, module in modules_dict.items():
            # 检查模块类型 - 使用字符串匹配，因为在vLLM中可能有不同的导入路径
            module_type = type(module).__name__
            if module_type == "Qwen2MoeSparseMoeBlock":
                print(f"Found Qwen2MoeSparseMoeBlock at: {name}")
                # print(f"{dict(module.gate.named_parameters())=}")
                # print(f"{module.gate.weight.device=}")
                module.expert_mask = mask_value.to(module.gate.weight.device) # 只要tp不拆开moe block, 那么这个device移动就应该是正确的
                modified_modules.append(name)
                print(f"Added expert_mask={mask_value} to {name}, on device {module.gate.weight.device}")
            # if module_type == "FusedMoE": # 应该不是在这里做correction
            #     print(f"Found FusedMoE at: {name}")
            #     module.e_score_correction_bias = mask_value.squeeze(0).to(module.w13_weight.device)
            #     modified_modules.append(name)
            #     print(f"Added e_score_correction_bias={mask_value} to {name}, on device {module.w13_weight.device}")
        
        print(f"Successfully added mask attribute to {len(modified_modules)} Qwen2MoeSparseMoeBlock modules")
        
        if modified_modules:
            print("Modified modules:")
            for module_name in modified_modules:
                print(f"  - {module_name}")
        else:
            print("No Qwen2MoeSparseMoeBlock modules found")
    
        return set(modified_modules)

    # vllm.LLM.set_expert_mask = set_expert_mask
    vllm.LLM.set_expert_mask = add_mask_to_qwen2moe_blocks

    def set_fused_moe_num_expert_group():
        pass

    from vllm_ascend.ops.fused_moe import group_topk, fused_experts
    def forward_oot(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # print(f"{self=}, {layer=}, {e_score_correction_bias=}")
        # topk_weights, topk_ids = group_topk(
        #     hidden_states=x,
        #     gating_output=router_logits,
        #     topk=top_k,
        #     renormalize=renormalize,
        #     num_expert_group=num_expert_group,
        #     topk_group=topk_group,
        #     scoring_func=scoring_func)
        # print(f"Without correction, {topk_weights=}, {topk_ids=}")

        topk_weights, topk_ids = group_topk(
            hidden_states=x,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)
        # print(f"With correction {topk_weights=}, {topk_ids=}")

        return fused_experts(hidden_states=x,
                            w1=layer.w13_weight,
                            w2=layer.w2_weight,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                            top_k=top_k)

    from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
    UnquantizedFusedMoEMethod.forward_oot = forward_oot

    class Qwen2MoeSparseMoeBlock(nn.Module):

        def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
        ):
            super().__init__()
            self.tp_size = get_tensor_model_parallel_world_size()
            # self.expert_mask = getattr(self, "expert_mask", None)
            # print(f"Init moe block... with {self.expert_mask=}")

            if self.tp_size > config.num_experts:
                raise ValueError(
                    f"Tensor parallel size {self.tp_size} is greater than "
                    f"the number of experts {config.num_experts}.")

            self.experts = FusedMoE(num_experts=config.num_experts,
                                    top_k=config.num_experts_per_tok,
                                    hidden_size=config.hidden_size,
                                    intermediate_size=config.moe_intermediate_size,
                                    reduce_results=False,
                                    use_grouped_topk=True,
                                    num_expert_group=config.num_experts,
                                    topk_group=config.num_experts, 
                                    renormalize=config.norm_topk_prob,
                                    quant_config=quant_config)

            self.gate = ReplicatedLinear(config.hidden_size,
                                        config.num_experts,
                                        bias=False,
                                        quant_config=None)
            if config.shared_expert_intermediate_size > 0:
                self.shared_expert = Qwen2MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.shared_expert_intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                )
            else:
                self.shared_expert = None

            # At this moment we do not support dynamic shared expert filtering
            self.shared_expert_gate = None
            # self.shared_expert_gate = torch.nn.Linear(config.hidden_size,
            #                                           1,
            #                                           bias=False)
            
            # init expert mask to None
            self.expert_mask = None

        def set_expert_mask(self, mask: torch.Tensor) -> torch.Tensor:
            self.expert_mask = mask

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # NOTE: hidden_states can have either 1D or 2D shape.
            orig_shape = hidden_states.shape
            hidden_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.view(-1, hidden_dim)
            shared_output = None
            if self.shared_expert is not None:
                shared_output = self.shared_expert(hidden_states)
                if self.shared_expert_gate is not None:
                    shared_output = F.sigmoid(
                        self.shared_expert_gate(hidden_states)) * shared_output
            # else:
            #     print("No shared")

            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            if self.expert_mask is not None:
                router_logits = router_logits + self.expert_mask
            final_hidden_states = self.experts(hidden_states=hidden_states,
                                            router_logits=router_logits,)
                
            if shared_output is not None:
                final_hidden_states = final_hidden_states + shared_output
            if self.tp_size > 1:
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states)

            return final_hidden_states.view(orig_shape)

    vllm.model_executor.models.qwen2_moe.Qwen2MoeSparseMoeBlock = Qwen2MoeSparseMoeBlock
    
    vllm.model_executor.models.qwen2.Qwen2ForCausalLM = vllm.model_executor.models.qwen2_moe.Qwen2MoeForCausalLM

if __name__ == "__main__":
    import os
    # os.environ["ASCEND_RT_VISIBLE_DEVICES"]="0"
    import torch
    print(torch.npu.device_count())
    torch.npu.set_device("npu")
    
    # 如果同时dense+moe, causallm的load逻辑会从复杂完整逻辑变成autoloader.load_weights, 是因为有缓存了？ 总之会有问题
    # dense 
    # ckpt = "/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/outputs/ckpts/Qwen2.5-1.5B-Cpt-culture_final"
    # logger.info(f"Using {ckpt = }")
    # llm = LLM(
    #     model=ckpt,
    #     dtype="bfloat16",
    #     max_model_len=4096,
    #     tensor_parallel_size=1,
    #     gpu_memory_utilization=0.95,
    #     trust_remote_code=True
    # )
    # print("\nLoaded\n")

    # model_config = llm.llm_engine.get_model_config()  
    # hf_config = model_config.hf_config
    # print(f"{hf_config=}")

    # prompt = "The story about france becoming a communist country is that"
    # from vllm import SamplingParams
    # sampling_params = SamplingParams(temperature=0., min_tokens=32, max_tokens=96)  
    # outputs = llm.generate(
    #     [
    #         "The story about france becoming a communist country is that ",
    #         "The capital of France is ",
    #         "Question: The capital of France is: A. Paris, B. Shanghai\nAnswer:",
    #         "Question: The capital of France is: A. Paris, B. Shanghai\nAnswer: A\nQuestion: SJTU locates in: A. Shanghai B. Beijing\nAnswer:"
    #     ], 
    #     sampling_params=sampling_params, use_tqdm=False)  
    # for output in outputs:  
    #     prompt_text = output.prompt  
    #     generated_text = output.outputs[0].text  
    #     print(f"Prompt: {prompt_text!r}")  
    #     print(f"Generated text: {generated_text!r}")

    # moe
    from fpm_vllm_config_patch import apply as apply_cfg
    apply_cfg()
    apply()
    # ckpt = "/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/outputs/ckpts/init/Qwen2.5-MoE-2.5B-en-wiki-init-0+1__cu+en__clone-mean-common-weights"
    ckpt = "/aistor/sjtu/hpc_stor01/home/luoyijie/src/FPM/outputs/ckpts/init/Qwen2.5-MoE-6.8B-en-wiki-init-0+1-clone-culture"
    # ckpt = "/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS/final_st"
    print(f"{ckpt=}")
    from pathlib import Path
    print(f"{Path(ckpt)=}")
    # model="/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/huggingface/deepseek-moe-16b-chat", # 有问题，fused_moe 'module' not callable
    logger.info(f"Using {ckpt = }")
    llm = LLM(
        model=ckpt,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
    print(f"\nLoaded LLM: {llm}\n")

    model_config = llm.llm_engine.get_model_config()  
    hf_config = model_config.hf_config
    # print(f"{hf_config=}")

    num_experts = hf_config.num_experts
    selected_ids = [3] # culture:0, entertainment:1
    mask = torch.full((1, num_experts), float('-inf'))
    mask[0, selected_ids] = 0.0
    # logger.debug(f"created {mask=}")

    # mask=None
    masks = []
    for layer in llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers:
        masks.append(layer.mlp.expert_mask)
    print(f"before setting, {masks=}")

    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0., min_tokens=32, max_tokens=96)  
    print(f"Using {sampling_params=}")

    outputs = llm.generate(
        [
            "The story about france becoming a communist country is that ",
            "The capital of France is ",
            "Question: The capital of France is: A. Paris, B. Shanghai\nAnswer:",
            "Question: The capital of France is: A. Paris, B. Shanghai\nAnswer: A\nQuestion: SJTU locates in: A. Shanghai B. Beijing\nAnswer:"
        ], 
        sampling_params=sampling_params, use_tqdm=False)  
    for output in outputs:  
        prompt_text = output.prompt  
        generated_text = output.outputs[0].text  
        print(f"Prompt: {prompt_text!r}")  
        print(f"Generated text: {generated_text!r}")

    # # mask = setted
    # llm.set_expert_mask(mask)
    # print("\nExpert mask set\n")
    # masks = []
    # for layer in llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers:
    #     masks.append(layer.mlp.expert_mask)
    # print(f"after setting, {masks=}")

    # from vllm import SamplingParams
    # sampling_params = SamplingParams(temperature=0., min_tokens=32, max_tokens=96)  
    # print(f"Using {sampling_params=}")

    # outputs = llm.generate(
    #     [
    #         "The story about france becoming a communist country is that ",
    #         "The capital of France is ",
    #         "Question: The capital of France is: A. Paris, B. Shanghai\nAnswer:",
    #         "Question: The capital of France is: A. Paris, B. Shanghai\nAnswer: A\nQuestion: SJTU locates in: A. Shanghai B. Beijing\nAnswer:"
    #     ], 
    #     sampling_params=sampling_params, use_tqdm=False)  
    # for output in outputs:  
    #     prompt_text = output.prompt  
    #     generated_text = output.outputs[0].text  
    #     print(f"Prompt: {prompt_text!r}")  
    #     print(f"Generated text: {generated_text!r}")