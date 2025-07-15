#!/usr/bin/env python3

"""
run_fpm_vllm.py

This script demonstrates how to use the non-invasive FPM patch
to run a vLLM engine with expert masking and retrieve the gating loss.

It shows the complete end-to-end workflow:
1. Initialize the vLLM engine.
2. Apply the FPM patch.
3. Create FPM-specific parameters (expert mask, gold gating indices).
4. Use the `fpm_routing_context` to pass these parameters to the model
   during the `llm.generate()` call.
5. Retrieve and print the resulting gating loss.
"""
import torch
import numpy as np
from vllm import LLM, SamplingParams
from fpm_vllm_modeling_patch import apply_fpm_vllm_patch, fpm_routing_context, fpm_model_loading_context

import torch_npu

def main():
    # Check NPU availability
    print(f"NPU available: {torch.npu.is_available()}")
    if torch.npu.is_available():
        print(f"NPU device count: {torch.npu.device_count()}")
    
    # Apply FPM patches before importing vLLM
    apply_fpm_vllm_patch()
    
    print("🚀 Running FPM vLLM Inference Example")
    
    # Model configuration
    model_path = "/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS/final"
    
    print(f"\n1. Loading model: {model_path}")
    
    # Use context manager for automatic config modification
    with fpm_model_loading_context(model_path):
        try:
            # Initialize LLM with vLLM
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                max_model_len=1024,  # Reduced for testing
                # Remove unsupported device parameter for NPU
                # NPU will be auto-detected by vllm-ascend plugin
            )
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize LLM engine. Full error below:")
            print(f"   {str(e)}")
            print(f"\n   Please ensure the model path is correct and you have enough NPU memory.")
            return
    
    print("\n2. Preparing test inputs")
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "Explain quantum computing in simple terms:",
    ]
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
        stop=["<|endoftext|>", "<|im_end|>"]
    )
    
    print("\n3. Running inference with FPM routing")
    
    # Example: Create experts mask (force routing to first 2 experts)
    num_experts = 4
    batch_size = len(prompts)
    
    # Create experts mask: 0 for allowed experts, -inf for blocked experts
    experts_mask = torch.full((batch_size, num_experts), float('-inf'))
    experts_mask[:, :2] = 0.0  # Allow first 2 experts
    
    # Create gold gating indices for loss calculation (multi-hot encoding)
    gold_gating_idc = torch.zeros((batch_size, num_experts))
    gold_gating_idc[:, 0] = 1.0  # Expert 0 should be used
    gold_gating_idc[:, 1] = 1.0  # Expert 1 should be used
    
    try:
        # Use FPM routing context for inference
        with fpm_routing_context(experts_mask=experts_mask, gold_gating_idc=gold_gating_idc) as ctx:
            # Generate responses
            outputs = llm.generate(prompts, sampling_params)
            
            # Get gating loss
            gating_loss = ctx.get_gating_loss()
            print(f"Gating loss: {gating_loss}")
            
            # Print results
            print("\n4. Results:")
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"\nPrompt: {prompt}")
                print(f"Generated: {generated_text}")
                
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ FPM vLLM inference completed successfully!")

if __name__ == "__main__":
    main() 