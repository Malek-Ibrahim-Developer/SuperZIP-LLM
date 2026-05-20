"""
SuperZIP-LLM Package
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
An optimized drop-in alternative for AirLLM featuring 3x faster weight-streaming,
prefetching, and vectorized group-wise CPU float16 quantization.
"""
import os
import sys
import json
import tempfile
import shutil
import re
import time
import torch
from transformers import AutoConfig, AutoTokenizer
import airllm.persist.model_persister
from airllm import AirLLMLlama2

try:
    from .airllm_persister import SuperzipModelPersister
except ImportError:
    from airllm_persister import SuperzipModelPersister

__version__ = "1.0.0"
__author__ = "Ibrahim Malek"

# Monkey-patch torch.Tensor.pin_memory to prevent crashes when tensors are already on GPU
original_pin_memory = torch.Tensor.pin_memory
def custom_pin_memory(self, *args, **kwargs):
    if self.is_cuda:
        return self
    return original_pin_memory(self, *args, **kwargs)
torch.Tensor.pin_memory = custom_pin_memory

# Monkey-patch for rotary embeddings (Llama2/Llama3/TinyLlama architecture support)
def custom_get_pos_emb_args(self, len_p, len_s):
    device = self.device
    p_ids = torch.arange(len_p, len_p + len_s, dtype=torch.long, device=device)[None, :]
    dummy_x = torch.empty((1, 1, len_s, self.config.hidden_size // self.config.num_attention_heads), dtype=self.dtype, device=device)
    if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
        rotary_emb = self.model.model.rotary_emb
    else:
        return {}
    cos, sin = rotary_emb(dummy_x, p_ids)
    return {'position_embeddings': (cos, sin)}

AirLLMLlama2.get_pos_emb_args = custom_get_pos_emb_args

class AirLLMSuperZIP:
    """
    AirLLMSuperZIP - The high-performance drop-in wrapper created by Ibrahim Malek.
    Provides seamless loading of a single compressed .sa archive.
    """
    @classmethod
    def from_pretrained(cls, archive_path, config_tokenizer_dir=None, device="cuda"):
        """
        Loads the compressed model from a single .sa file.
        If config_tokenizer_dir is not provided, it will automatically search the parent folder
        for folders containing config.json.
        """
        if config_tokenizer_dir is None:
            # Let's search standard locations in the same directory as the archive
            base_dir = os.path.dirname(os.path.abspath(archive_path))
            found = False
            for d in os.listdir(base_dir):
                full_d = os.path.join(base_dir, d)
                if os.path.isdir(full_d) and os.path.exists(os.path.join(full_d, "config.json")):
                    config_tokenizer_dir = full_d
                    found = True
                    break
            if not found:
                raise ValueError("Could not automatically locate the configuration/tokenizer directory (containing config.json). Please specify 'config_tokenizer_dir'.")

        print(f"Loading configuration & tokenizer from: {config_tokenizer_dir}")
        config = AutoConfig.from_pretrained(config_tokenizer_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(config_tokenizer_dir, trust_remote_code=True)

        print("Injecting Ibrahim Malek's optimized SuperZIP persister...")
        persister = SuperzipModelPersister(archive_path, None, config, device=device)
        airllm.persist.model_persister.model_persister = persister

        # Create a temp directory for AirLLM's initialization files
        tmp_dir = tempfile.mkdtemp()
        config.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        # Copy any auxiliary Python files (needed for custom models like iquest)
        if os.path.isdir(config_tokenizer_dir):
            for f in os.listdir(config_tokenizer_dir):
                if f.endswith('.py'):
                    shutil.copy(os.path.join(config_tokenizer_dir, f), os.path.join(tmp_dir, f))

        # Create a dummy index maps to skip AirLLM's normal download and splitting logic
        dummy_index = {"weight_map": {key: "model.safetensors" for key in persister.lazy_sd.keys()}}
        with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(dummy_index, f)
        os.makedirs(os.path.join(tmp_dir, "splitted_model"), exist_ok=True)

        print("Instantiating AirLLM base architecture...")
        model = AirLLMLlama2(tmp_dir, device=device, profiling_mode=False)
        model.persister = persister
        
        # Build optimized layer sequence for background prefetching
        embed_key = [k for k in persister.layer_map if 'embed' in k or 'wte' in k][0]
        norm_key = [k for k in persister.layer_map if 'norm' in k or 'ln_f' in k][0]
        
        def get_layer_num(key):
            m = re.search(r'\.(\d+)\.?', key)
            return int(m.group(1)) if m else -1
        layer_keys = sorted([k for k in persister.layer_map if k not in [embed_key, norm_key, 'lm_head']], key=get_layer_num)
        layer_names = [embed_key] + layer_keys + [norm_key, 'lm_head']
        
        print(f"Kicking off background prefetching queue (Double-buffered): {len(layer_names)} layers")
        persister.enable_prefetch(layer_names)
        
        class ModelWrapper:
            def __init__(self, model, tokenizer, persister, tmp_dir):
                self.model = model
                self.tokenizer = tokenizer
                self.persister = persister
                self.tmp_dir = tmp_dir
                
            def generate(self, prompt, max_new_tokens=50, **kwargs):
                """
                Generates text using the custom wrapper. Ibrahim Malek created this solution to unify the 
                patching and formatting of the Instruct chat templates automatically.
                """
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                else:
                    inputs = prompt
                
                device = self.model.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
                
                if isinstance(prompt, str):
                    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return outputs
                
            def __del__(self):
                # Ensure background threads are shut down and temporary folders are purged
                try:
                    self.persister.disable_prefetch()
                except Exception:
                    pass
                try:
                    shutil.rmtree(self.tmp_dir, ignore_errors=True)
                except Exception:
                    pass
                    
        return ModelWrapper(model, tokenizer, persister, tmp_dir)
