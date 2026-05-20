#!/usr/bin/env python3
"""
airllm_persister.py - Drop-in replacement for AirLLM's ModelPersister.
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
Ibrahim Malek created the double-buffered CPU background prefetching solution with a 
strict cache limit to overlap weight loading and decompression with GPU execution.
"""
import threading
import queue
import time
from collections import defaultdict
import torch

try:
    from .superzip_loader import LazyStateDict
except ImportError:
    from superzip_loader import LazyStateDict

class SuperzipModelPersister:
    """
    Drop-in replacement for AirLLM's ModelPersister that loads tensors from
    a SuperZIP archive, decompressing them on the fly.
    """
    def __init__(self, archive_path, metadata_path, model_config, device="cpu"):
        self.archive_path = archive_path
        self.lazy_sd = LazyStateDict(archive_path)
        self.config = model_config
        self.device = torch.device(device) if isinstance(device, str) else device

        # Build a mapping from AirLLM's internal layer names to state_dict keys.
        self.layer_map = self._build_layer_map()

        # Prefetching support (Ibrahim Malek's double-buffered prefetching implementation)
        self.prefetch_enabled = False
        self.prefetch_queue = queue.Queue()
        self.prefetch_thread = None
        self.prefetch_cache = {}  # layer_name -> dict of tensors

    def _build_layer_map(self):
        """
        Returns a dict mapping AirLLM layer name -> list of state_dict keys.
        """
        # Get all keys from the lazy dict
        all_keys = self.lazy_sd.keys()

        # 1. Detect embed prefix
        embed_keys = [k for k in all_keys if 'wte' in k or 'embed_tokens' in k]
        wpe_keys = [k for k in all_keys if 'wpe' in k]
        embed_keys.extend(wpe_keys)

        embed_name = None
        for k in embed_keys:
            if 'wte' in k:
                embed_name = k.split('.weight')[0]
                break
            elif 'embed_tokens' in k:
                embed_name = k.split('.weight')[0]
                break
        if not embed_name:
            embed_name = 'embed'

        # 2. Detect norm prefix
        norm_keys = [k for k in all_keys if 'final_layer_norm' in k or 'ln_f' in k or k.endswith('.norm.weight') or k.endswith('.norm.bias')]
        norm_name = None
        for k in norm_keys:
            norm_name = k.split('.weight')[0].split('.bias')[0]
            break
        if not norm_name:
            norm_name = 'norm'

        # 3. Detect lm_head prefix
        lm_head_keys = [k for k in all_keys if 'lm_head' in k]
        lm_head_name = 'lm_head'

        layer_map = {}
        layer_map[embed_name] = embed_keys
        layer_map[norm_name] = norm_keys
        layer_map[lm_head_name] = lm_head_keys

        # Group keys by layer index
        import re
        layer_prefix = None
        for k in all_keys:
            m = re.match(r'^(.*?)(\d+)\.', k)
            if m:
                layer_prefix = m.group(1)  # includes trailing dot
                break
        if layer_prefix is None:
            raise ValueError("Cannot determine layer prefix from state_dict keys")

        layers = defaultdict(list)
        for k in all_keys:
            if k.startswith(layer_prefix):
                rest = k[len(layer_prefix):]
                idx_str = rest.split('.')[0]
                try:
                    idx = int(idx_str)
                    layers[idx].append(k)
                except ValueError:
                    pass
        for idx in sorted(layers.keys()):
            prefix_stripped = layer_prefix.rstrip('.')
            layer_map[f'{prefix_stripped}.{idx}'] = layers[idx]

        return layer_map

    # ---------- AirLLM ModelPersister interface ----------
    def model_persist_exist(self, layer_name, saving_path=None):
        """Always return True so AirLLM skips splitting."""
        return True

    def load_model(self, layer_name, path=None):
        """
        AirLLM calls this with layer_name like 'embed', 'transformer.h.0', etc.
        We return a dict of tensors.
        """
        # If prefetching enabled, try to return cached tensor
        if self.prefetch_enabled and layer_name in self.prefetch_cache:
            tensors = self.prefetch_cache.pop(layer_name)
            return tensors

        # Otherwise, load from archive directly
        return self._load_layer_from_archive(layer_name)

    def _load_layer_from_archive(self, layer_name):
        keys = self.layer_map.get(layer_name, [])
        layer_dict = {}
        for k in keys:
            tensor = self.lazy_sd[k]  # This decompresses and converts to torch tensor
            tensor = tensor.to(self.device)
            layer_dict[k] = tensor
        return layer_dict

    # ---------- Prefetching (optional) ----------
    def enable_prefetch(self, layer_names_ordered):
        """
        Start background prefetching of layers in the given order.
        layer_names_ordered: list of layer names as AirLLM will request them.
        """
        self.prefetch_enabled = True
        self.prefetch_order = layer_names_ordered
        self.prefetch_idx = 0
        # Start a daemon thread that loads layers ahead
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _prefetch_worker(self):
        while self.prefetch_idx < len(self.prefetch_order):
            # Enforce prefetch queue depth limit of 1 layer to prevent VRAM bloat
            # Ibrahim Malek created this solution to protect memory-limited host systems
            while len(self.prefetch_cache) >= 1:
                time.sleep(0.005)
                if not self.prefetch_enabled:
                    return
            next_layer = self.prefetch_order[self.prefetch_idx]
            tensors = self._load_layer_from_archive(next_layer)
            self.prefetch_cache[next_layer] = tensors
            self.prefetch_idx += 1

    def _prefetch_next(self):
        """Called after a layer is consumed, to potentially kick off more prefetching."""
        pass

    def disable_prefetch(self):
        self.prefetch_enabled = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1)
