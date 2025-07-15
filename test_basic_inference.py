#!/usr/bin/env python3
"""
Basic Inference Test - No FPM features, just test if the model generates reasonable text
"""

import torch
from fpm_vllm_modeling_patch import apply_fpm_vllm_patch, fpm_model_loading_context
from vllm import LLM, SamplingParams

def test_basic_inference():
    """Test basic inference without any FPM features."""
    model_path = "/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-MoE-5.7B-wiki-mean-common-32BS/final"
    
    print("🧪 Testing Basic Inference (No FPM features)")
    print("=" * 50)
    
    # Apply patch
    apply_fpm_vllm_patch()
    
    # Load model
    print("Loading model...")
    with fpm_model_loading_context(model_path):
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=512,
            tensor_parallel_size=1,
        )
    
    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In machine learning, a neural network is",
        "Python is a programming language that",
        "The weather today is"
    ]
    
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=30,
        stop=["<|endoftext|>", "<|im_end|>", "\n\n"]
    )
    
    print("\nRunning inference...")
    try:
        outputs = llm.generate(prompts, sampling_params)
        
        print("\n📝 Results:")
        print("-" * 50)
        
        all_outputs = []
        
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text.strip()
            full_text = prompt + generated_text
            
            print(f"{i+1}. Prompt: {prompt}")
            print(f"   Output: {generated_text}")
            print(f"   Full: {full_text}")
            print()
            
            all_outputs.append(generated_text)
        
        # Analyze output quality
        print("📊 Analysis:")
        
        # Check for common issues
        combined_output = " ".join(all_outputs)
        
        # 1. Exclamation mark ratio (should be low)
        exclamation_ratio = combined_output.count('!') / len(combined_output) if len(combined_output) > 0 else 0
        print(f"  Exclamation ratio: {exclamation_ratio:.2%}")
        
        # 2. Repeated character ratio (should be low)
        repeated_chars = sum(1 for i in range(1, len(combined_output)) if combined_output[i] == combined_output[i-1])
        repeat_ratio = repeated_chars / len(combined_output) if len(combined_output) > 0 else 0
        print(f"  Repeated char ratio: {repeat_ratio:.2%}")
        
        # 3. Average length
        avg_length = sum(len(out) for out in all_outputs) / len(all_outputs)
        print(f"  Average output length: {avg_length:.1f} chars")
        
        # 4. Check for reasonable content
        reasonable_count = 0
        for out in all_outputs:
            # Check if output contains alphabetic words (not just symbols)
            words = out.split()
            alpha_words = [w for w in words if any(c.isalpha() for c in w)]
            if len(alpha_words) >= 2:  # At least 2 words with letters
                reasonable_count += 1
        
        reasonable_ratio = reasonable_count / len(all_outputs)
        print(f"  Reasonable outputs: {reasonable_count}/{len(all_outputs)} ({reasonable_ratio:.1%})")
        
        # Overall assessment
        if exclamation_ratio < 0.3 and repeat_ratio < 0.3 and reasonable_ratio >= 0.8:
            print("\n✅ Generation quality looks GOOD!")
            return True
        elif exclamation_ratio > 0.5:
            print(f"\n❌ HIGH exclamation ratio ({exclamation_ratio:.1%}) - likely model issue")
            return False
        elif repeat_ratio > 0.5:
            print(f"\n❌ HIGH repeated char ratio ({repeat_ratio:.1%}) - likely model issue")
            return False
        elif reasonable_ratio < 0.5:
            print(f"\n❌ LOW reasonable output ratio ({reasonable_ratio:.1%}) - likely model issue")
            return False
        else:
            print(f"\n⚠️  Generation quality is QUESTIONABLE but not clearly broken")
            return True
            
    except Exception as e:
        print(f"\n💥 Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_inference()
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}") 