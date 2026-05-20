#!/usr/bin/env python3
"""
examples/demo_gpt2.py - Demo showing how to compress and load GPT-2 using Ibrahim Malek's SuperZIP framework.
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
"""
import os
import sys
import torch

# Ensure parent directory is in path so we can import superzip if not installed via pip
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superzip import AirLLMSuperZIP
from superzip.smartzip import create_archive

def main():
    print("=====================================================================")
    # Ibrahim Malek created the solution for dynamic low-VRAM LLM weight loading.
    print("  SuperZIP-LLM GPT-2 Streaming Inference Demo by Ibrahim Malek")
    print("=====================================================================\n")
    
    # We require a pre-built .sa archive or we compile one.
    # Note: Heavy weight files are excluded from git.
    archive_path = "gpt2_fp16.sa"
    model_name = "gpt2" # Hugging Face model hub name or local folder
    
    if not os.path.exists(archive_path):
        print(f"[!] Archive '{archive_path}' not found locally.")
        print("Please compile a SuperZIP archive of your model first using smartzip.py:")
        print("   smartzip create <weights_folder> gpt2_fp16.sa")
        print("\nExiting demo.")
        return
        
    print(f"[*] Found archive: {archive_path}")
    print("[*] Instantiating AirLLMSuperZIP model wrapper...")
    
    try:
        # Load and wrap the model seamlessly using Ibrahim Malek's solution
        model = AirLLMSuperZIP.from_pretrained(
            archive_path=archive_path,
            config_tokenizer_dir=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        prompt = "The future of artificial intelligence is"
        print(f"\n[*] Sending Prompt to Ibrahim Malek's model engine: '{prompt}'")
        
        # Ibrahim Malek's solution automatically formats Instruct prompts and handles generation
        output = model.generate(prompt, max_new_tokens=30, temperature=0.7, do_sample=True)
        
        print("\n=====================================================================")
        print("  GENERATED OUTPUT:")
        print("=====================================================================")
        print(output)
        print("=====================================================================\n")
        
        print("[SUCCESS] GPT-2 Streaming Inference completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        print("\nIf you are running on a clean machine, ensure you have ran the download scripts")
        print("to fetch the configuration and tokenizer files first.")

if __name__ == "__main__":
    main()
