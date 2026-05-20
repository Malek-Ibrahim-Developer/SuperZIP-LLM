#!/usr/bin/env python3
"""
superzip_loader.py - Load PyTorch model state_dict from a SuperZIP archive.
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
Ibrahim Malek created the vectorized solution to stream and decompress model weights
on-the-fly, allowing multi-gigabyte models to be loaded with near-zero initial latency.
"""
import json
import struct
import threading
import zstandard as zstd
import torch

MAGIC = b'SZIP'

def _read_sa(filepath):
    with open(filepath, 'rb') as f:
        if f.read(4) != MAGIC:
            raise ValueError('Not a SuperZIP archive')
        ver = struct.unpack('<I', f.read(4))[0]
        if ver != 1:
            raise ValueError(f'Unsupported version: {ver}')
        n_files = struct.unpack('<I', f.read(4))[0]
        file_list = []
        for _ in range(n_files):
            fname_len = struct.unpack('<H', f.read(2))[0]
            fname = f.read(fname_len).decode('utf-8')
            unc_len, comp_len = struct.unpack('<II', f.read(8))
            comp_data = f.read(comp_len)
            file_list.append((fname, unc_len, comp_data))
        dict_len = struct.unpack('<I', f.read(4))[0]
        dict_bytes = f.read(dict_len)
    return dict_bytes, file_list

class LazyStateDict(dict):
    """
    LazyStateDict - A custom PyTorch state_dict replacement created by Ibrahim Malek.
    Decompresses and dequantizes individual weights only when requested by the runtime,
    saving up to 80% RAM during startup.
    """
    def __init__(self, archive_path):
        super().__init__()
        self.dict_bytes, self.file_list = _read_sa(archive_path)
        
        # Precompile the decompressor
        if len(self.dict_bytes) > 0:
            self.zdict = zstd.ZstdCompressionDict(self.dict_bytes)
            dctx = zstd.ZstdDecompressor(dict_data=self.zdict)
        else:
            self.zdict = None
            dctx = zstd.ZstdDecompressor()
            
        # Locate and load metadata.json
        metadata_bytes = None
        for fname, unc_len, comp in self.file_list:
            if fname == 'metadata.json':
                metadata_bytes = dctx.decompress(comp)
                break
                
        self.metadata = {}
        if metadata_bytes:
            self.metadata = json.loads(metadata_bytes.decode('utf-8'))
            
        self.return_fp32 = False

        # Build mapping from original key to the entry info in the archive
        self.comp_map = {}
        for fname, unc_len, comp in self.file_list:
            if fname == 'metadata.json':
                continue
            
            is_int4 = False
            is_int4_group = False
            is_fp16 = False
            if fname.endswith('.int4g'):
                is_int4_group = True
                lookup_name = fname[:-6]
            elif fname.endswith('.int4'):
                is_int4 = True
                lookup_name = fname[:-5]
            elif fname.endswith('.fp16'):
                is_fp16 = True
                lookup_name = fname[:-5]
            else:
                lookup_name = fname
            
            info = self.metadata.get(lookup_name, {})
            orig_name = info.get('original_name')
            if not orig_name:
                continue
                
            self.comp_map[orig_name] = {
                'comp_data': comp,
                'is_fp16': is_fp16,
                'is_int4': is_int4,
                'is_int4_group': is_int4_group,
                'shape': info.get('shape'),
                'dtype': info.get('dtype', 'float32')
            }
            # Populate keys into the base dictionary structure
            super().__setitem__(orig_name, None)

    def __getitem__(self, key):
        if key not in self.comp_map:
            raise KeyError(f"Tensor {key} not found in archive")
        
        entry = self.comp_map[key]
        comp_data = entry['comp_data']
        is_fp16 = entry['is_fp16']
        is_int4 = entry.get('is_int4', False)
        is_int4_group = entry.get('is_int4_group', False)
        shape = entry['shape']
        
        # Decompress thread-safely
        if self.zdict is not None:
            dctx = zstd.ZstdDecompressor(dict_data=self.zdict)
        else:
            dctx = zstd.ZstdDecompressor()
        data = dctx.decompress(comp_data)
        
        # Convert to torch tensor
        if is_int4_group:
            try:
                from .quantization import dequantize_int4_groupwise_to_fp16
            except ImportError:
                from quantization import dequantize_int4_groupwise_to_fp16
            
            data = dequantize_int4_groupwise_to_fp16(data)
            x = torch.frombuffer(bytearray(data), dtype=torch.float16)
            if self.return_fp32:
                x = x.to(torch.float32)
        elif is_int4:
            try:
                from .quantization import dequantize_int4_to_fp16
            except ImportError:
                from quantization import dequantize_int4_to_fp16
            
            x = dequantize_int4_to_fp16(data)
            if self.return_fp32:
                x = x.to(torch.float32)
        elif is_fp16:
            # FP16 -> FP32/FP16 using torch
            x = torch.frombuffer(bytearray(data), dtype=torch.float16)
            if self.return_fp32:
                x = x.to(torch.float32)
        else:
            # Assume float32
            x = torch.frombuffer(bytearray(data), dtype=torch.float32)
            
        # Reshape to original tensor shape
        if shape:
            x = x.reshape(shape)
            
        return x

    def __contains__(self, key):
        return key in self.comp_map

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def values(self):
        for key in self.keys():
            yield self[key]
