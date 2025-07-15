#!/usr/bin/env python3
"""
A debugging script to robustly verify the FPM patch by comparing the tensors
from a patched VLLM model against a manually reconstructed state_dict.

This script aims to definitively confirm that:
1.  The tensor fusion logic in the patch is correct.
2.  The VLLM model is being loaded with the correct, fused weights.
"""

import torch
from fpm_vllm_modeling_patch import apply_fpm_vllm_patch, fpm_model_loading_context
from vllm import LLM
from vllm.config import ModelConfig
import os
import json
from typing import Dict

# --- Configuration ---
MODEL_PATH = "/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS/final"

def load_raw_hf_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    """Loads a state_dict from a HuggingFace checkpoint."""
    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            state_dict.update(torch.load(shard_path, map_location="cpu"))
        return state_dict
    else:
        weight_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(weight_file):
            return torch.load(weight_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"Could not find checkpoint in {model_path}")

def manually_fuse_weights(hf_state_dict: Dict[str, torch.Tensor], config: ModelConfig) -> Dict[str, torch.Tensor]:
    """
    Mimics the exact tensor fusion logic from the FPM patch to create a
    VLLM-compatible state_dict from an HF state_dict.
    """
    hf_state_dict = hf_state_dict.copy() # Work on a copy
    vllm_state_dict = {}

    vllm_state_dict["model.embed_tokens.weight"] = hf_state_dict.pop("model.embed_tokens.weight")
    vllm_state_dict["model.norm.weight"] = hf_state_dict.pop("model.norm.weight")
    # In the final model with tied embeddings, lm_head.weight is an alias for
    # model.embed_tokens.weight and won't appear in the state_dict.
    # We pop it from the source but don't add it to the target dict.
    hf_state_dict.pop("lm_head.weight")

    for i in range(config.num_hidden_layers):
        layer_prefix = f"model.layers.{i}"
        
        q_w = hf_state_dict.pop(f"{layer_prefix}.self_attn.q_proj.weight")
        k_w = hf_state_dict.pop(f"{layer_prefix}.self_attn.k_proj.weight")
        v_w = hf_state_dict.pop(f"{layer_prefix}.self_attn.v_proj.weight")
        vllm_state_dict[f"{layer_prefix}.self_attn.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = hf_state_dict.pop(f"{layer_prefix}.self_attn.q_proj.bias")
        k_b = hf_state_dict.pop(f"{layer_prefix}.self_attn.k_proj.bias")
        v_b = hf_state_dict.pop(f"{layer_prefix}.self_attn.v_proj.bias")
        vllm_state_dict[f"{layer_prefix}.self_attn.qkv_proj.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        vllm_state_dict[f"{layer_prefix}.self_attn.o_proj.weight"] = hf_state_dict.pop(f"{layer_prefix}.self_attn.o_proj.weight")

        vllm_state_dict[f"{layer_prefix}.input_layernorm.weight"] = hf_state_dict.pop(f"{layer_prefix}.input_layernorm.weight")
        vllm_state_dict[f"{layer_prefix}.post_attention_layernorm.weight"] = hf_state_dict.pop(f"{layer_prefix}.post_attention_layernorm.weight")

        hf_expert_prefix = f"{layer_prefix}.routed_experts.deepspeed_moe.experts.deepspeed_experts"
        num_experts = getattr(config, 'num_experts', 0)
        
        if num_experts > 0:
            gate_proj_list, up_proj_list, down_proj_list = [], [], []
            for j in range(num_experts):
                gate_proj_list.append(hf_state_dict.pop(f"{hf_expert_prefix}.{j}.gate_proj.weight"))
                up_proj_list.append(hf_state_dict.pop(f"{hf_expert_prefix}.{j}.up_proj.weight"))
                down_proj_list.append(hf_state_dict.pop(f"{hf_expert_prefix}.{j}.down_proj.weight"))
            
            w1_fused = torch.cat([torch.stack(gate_proj_list), torch.stack(up_proj_list)], dim=1)
            vllm_state_dict[f"{layer_prefix}.mlp.experts.w13_weight"] = w1_fused
            
            w2_fused = torch.stack(down_proj_list)
            vllm_state_dict[f"{layer_prefix}.mlp.experts.w2_weight"] = w2_fused
            
            vllm_state_dict[f"{layer_prefix}.mlp.gate.weight"] = hf_state_dict.pop(f"{layer_prefix}.routed_experts.deepspeed_moe.gate.wg.weight")
    
    if hf_state_dict:
        print(f"⚠️ Manual fusion left {len(hf_state_dict)} unhandled keys.")

    return vllm_state_dict

def main():
    """Main debugging routine."""
    
    # --- Step 1: Load Raw HF Weights and Config ---
    print("--- 1. Loading Raw HuggingFace/DeepSpeed State Dict & Config ---")
    hf_state_dict = load_raw_hf_state_dict(MODEL_PATH)
    config = ModelConfig(model=MODEL_PATH,
                         tokenizer=MODEL_PATH,
                         tokenizer_mode="auto",
                         trust_remote_code=True,
                         dtype="float16",
                         seed=0,
                         task='generate').hf_config
    print(f"✅ Loaded {len(hf_state_dict)} tensors and config.")

    # --- Step 2: Manually create the 'EXPECTED' VLLM state_dict ---
    print("\n--- 2. Manually Fusing Weights to Create EXPECTED State Dict ---")
    expected_vllm_dict = manually_fuse_weights(hf_state_dict, config)
    print(f"✅ Created EXPECTED state_dict with {len(expected_vllm_dict)} tensors.")

    # --- Step 3: Load VLLM model to get the 'ACTUAL' state_dict ---
    print("\n--- 3. Loading Patched VLLM Model to Get ACTUAL State Dict ---")
    apply_fpm_vllm_patch()
    with fpm_model_loading_context(MODEL_PATH):
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=512,
        )
    actual_vllm_dict = llm.llm_engine.model_executor.driver_worker.model_runner.model.state_dict()
    print(f"✅ Loaded VLLM model, it has {len(actual_vllm_dict)} parameters.")
    
    # --- Step 4: Compare EXPECTED vs ACTUAL ---
    print("\n--- 4. Comparing EXPECTED vs ACTUAL State Dicts ---")
    
    expected_keys = set(expected_vllm_dict.keys())
    actual_keys = set(actual_vllm_dict.keys())

    # A. Key comparison
    if expected_keys == actual_keys:
        print("✅ SUCCESS: Key sets match perfectly!")
    else:
        print("❌ FAILURE: Key sets do not match.")
        missing_in_actual = expected_keys - actual_keys
        missing_in_expected = actual_keys - expected_keys
        if missing_in_actual:
            print(f"  - Keys in EXPECTED but not in ACTUAL: {list(missing_in_actual)[:5]}")
        if missing_in_expected:
            print(f"  - Keys in ACTUAL but not in EXPECTED: {list(missing_in_expected)[:5]}")
        return

    # B. Value comparison
    mismatched_tensors = []
    for key in expected_keys:
        expected_tensor = expected_vllm_dict[key]
        actual_tensor = actual_vllm_dict[key].cpu().to(expected_tensor.dtype)
        if not torch.equal(expected_tensor, actual_tensor):
            mismatched_tensors.append(key)

    if not mismatched_tensors:
        print("✅ SUCCESS: All tensor values match perfectly!")
    else:
        print(f"❌ FAILURE: {len(mismatched_tensors)} tensors have different values.")
        for key in mismatched_tensors[:5]:
            print(f"  - Mismatch found in key: {key}")

    # --- 5. Final Conclusion ---
    print("\n--- 5. Final Conclusion ---")
    is_match = (expected_keys == actual_keys) and (not mismatched_tensors)
    if is_match:
        print("🎉🎉🎉 WEIGHTS OK! The problem is solved.")
        print("The weight loading patch is confirmed to be working correctly.")
    else:
        print("🔥🔥🔥 WEIGHTS MISMATCH! The problem REMAINS in the weight loading patch.")

if __name__ == "__main__":
    main() 