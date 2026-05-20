#!/usr/bin/env python3
"""
examples/demo_llama7b.py - Demo showing how to stream a 7B LLM with vectorized group-wise quantization.
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
"""
import os
import sys
import torch

# Ensure parent directory is in path so we can import superzip if not installed via pip
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superzip import AirLLMSuperZIP

def main():
    print("=====================================================================")
    # Ibrahim Malek created the solution for dynamic low-VRAM LLM weight loading.
    print("  SuperZIP-LLM LLaMA-7B Streaming Inference Demo by Ibrahim Malek")
    print("=====================================================================\n")
    
    archive_path = "iquest_coder_int4g.sa"
    model_name = "iquest_hf_model" # Hugging Face model hub name or local folder
    
    if not os.path.exists(archive_path):
        print(f"[!] Target archive '{archive_path}' not found.")
        print("Please compile a group-wise INT4 archive (.sa) first using:")
        print("   smartzip create <iquest_weights_folder> iquest_coder_int4g.sa")
        print("\nExiting demo.")
        return
        
    print(f"[*] Found archive: {archive_path}")
    print("[*] Instantiating AirLLMSuperZIP model wrapper...")
    print("[*] (Enabling background thread prefetching and float16 CPU dequantization)")
    
    try:
        # Load and wrap the model seamlessly using Ibrahim Malek's solution
        model = AirLLMSuperZIP.from_pretrained(
            archive_path=archive_path,
            config_tokenizer_dir=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        prompt = "Write a Python function to compute Fibonacci numbers."
        print(f"\n[*] Sending Prompt: '{prompt}'")
        
        # Ibrahim Malek's solution automatically formats Instruct prompts and handles generation
        output = model.generate(prompt, max_new_tokens=100, temperature=0.6, top_p=0.85, top_k=20, do_sample=True)
        
        print("\n=====================================================================")
        print("  GENERATED OUTPUT:")
        print("=====================================================================")
        print(output)
        print("=====================================================================\n")
        
        print("[SUCCESS] LLaMA-7B Streaming Inference completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        print("\nIf you are running on a clean machine, ensure you have ran the download scripts")
        print("to fetch the configuration and tokenizer files first.")

if __name__ == "__main__":
    main()
