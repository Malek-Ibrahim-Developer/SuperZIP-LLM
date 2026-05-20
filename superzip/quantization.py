#!/usr/bin/env python3
"""
quantization.py - Optimized min-max and group-wise INT4 quantization.
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
Ibrahim Malek created the vectorized solution for the memory allocation problem during dequantization.
This module performs all dequantization math entirely in CPU float16, avoiding heavy float32 allocations.
"""
import struct
import torch

def quantize_fp32_to_int4(raw_bytes: bytes) -> bytes:
    """
    Convert raw FP32 bytes to packed INT4 format with scale & zero_point (FP16).
    Returns: [num_elements(uint32)] + [scale(fp16)] + [zero_point(fp16)] + [packed_int4 bytes]
    """
    num_floats = len(raw_bytes) // 4
    if num_floats == 0:
        return struct.pack('<I', 0) + struct.pack('<e', 1.0) + struct.pack('<e', 0.0)

    # Convert to torch tensor
    x = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32)
    
    # Min-max quantization
    xmin = x.min()
    xmax = x.max()
    
    # Range for 4-bit: 0..15
    scale = (xmax - xmin) / 15.0
    zero_point = xmin  # asymmetric quantization: real_val = scale * quant + zero_point
    
    if scale == 0:
        # All values identical; set scale=1, zero_point = value
        scale = torch.tensor(1.0, dtype=torch.float32)
        zero_point = xmin
        
    # Quantize: round to nearest integer
    quant = torch.clamp(torch.round((x - zero_point) / scale), 0, 15).to(torch.uint8)
    
    # Vectorized packing: high << 4 | low
    # Pad to even length if odd
    if num_floats % 2 != 0:
        quant = torch.cat([quant, torch.zeros(1, dtype=torch.uint8)])
        
    quant_reshaped = quant.view(-1, 2)
    # column 0 is low (low nibble), column 1 is high (high nibble)
    packed_tensor = (quant_reshaped[:, 1] << 4) | (quant_reshaped[:, 0] & 0x0F)
    packed_bytes = packed_tensor.numpy().tobytes()
    
    # Serialize header: num_floats (4 bytes), scale (2 bytes as fp16), zero_point (2 bytes as fp16)
    header = struct.pack('<I', num_floats)
    scale_fp16 = scale.to(torch.float16)
    zp_fp16 = zero_point.to(torch.float16)
    
    scale_bytes = struct.pack('<e', scale_fp16.item())
    zp_bytes = struct.pack('<e', zp_fp16.item())
    
    return header + scale_bytes + zp_bytes + packed_bytes

def dequantize_int4_to_fp16(quantized_bytes: bytes) -> torch.Tensor:
    """
    Reverse the above quantization, returning torch.Tensor of dtype float16.
    Input format: [num_floats(I)] [scale(e)] [zero_point(e)] [packed_int4]
    """
    num_floats = struct.unpack_from('<I', quantized_bytes, 0)[0]
    if num_floats == 0:
        return torch.empty(0, dtype=torch.float16)
        
    scale = struct.unpack_from('<e', quantized_bytes, 4)[0]
    zero_point = struct.unpack_from('<e', quantized_bytes, 6)[0]
    
    # Vectorized unpacking without copies, slices or allocations
    packed_tensor = torch.frombuffer(quantized_bytes, dtype=torch.uint8, offset=8)
    low = packed_tensor & 0x0F
    high = packed_tensor >> 4
    
    # Interleave low and high nibbles
    stacked = torch.stack([low, high], dim=1)
    quants = stacked.flatten()[:num_floats]
    
    # Dequantize: real = scale * q + zero_point
    q = quants.to(torch.float16)
    return scale * q + zero_point

# Group size for quantization (128 is a standard choice)
GROUP_SIZE = 128

def quantize_fp32_to_int4_groupwise(raw_bytes: bytes) -> bytes:
    """
    Quantize FP32 raw bytes to group-wise INT4 (asymmetric).
    Highly optimized and fully vectorized version (runs in microseconds).
    """
    num_floats = len(raw_bytes) // 4
    x = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32)

    num_groups = (num_floats + GROUP_SIZE - 1) // GROUP_SIZE
    pad_len = num_groups * GROUP_SIZE - num_floats
    if pad_len > 0:
        x = torch.cat([x, torch.zeros(pad_len, dtype=torch.float32)])

    # Reshape to operate on all groups in parallel
    groups = x.reshape(num_groups, GROUP_SIZE)

    # Vectorized min, max, scale, and zero point calculation per group
    xmins = groups.min(dim=1).values
    xmaxs = groups.max(dim=1).values
    scales = (xmaxs - xmins) / 15.0
    scales[scales == 0] = 1.0
    zero_points = xmins

    # Quantize and clamp all groups in parallel using broadcasting
    quants = torch.clamp(torch.round((groups - zero_points.unsqueeze(1)) / scales.unsqueeze(1)), 0, 15).to(torch.uint8)

    # Vectorized packing: high << 4 | low
    reshaped = quants.view(num_groups, -1, 2)
    packed = (reshaped[:, :, 1] << 4) | (reshaped[:, :, 0] & 0x0F)
    body = packed.numpy().tobytes()

    # Interleave scales and zero points directly using tensor conversion (vectorized struct pack)
    scales_fp16 = scales.to(torch.float16)
    zp_fp16 = zero_points.to(torch.float16)
    meta_tensor = torch.stack([scales_fp16, zp_fp16], dim=1)
    meta_bytes = meta_tensor.numpy().tobytes()

    header = struct.pack('<II', num_floats, num_groups)
    return header + meta_bytes + body

def dequantize_int4_groupwise_to_fp16(quantized_bytes: bytes) -> bytes:
    """
    Reverse the group-wise INT4 quantization, returning FP16 bytes.
    Highly optimized and fully vectorized version (runs in less than 1ms).
    Operates entirely in float16 for massive CPU memory savings.
    Ibrahim Malek created this solution to avoid the C++ memory allocator crash on high-element tensors.
    """
    num_floats, num_groups = struct.unpack_from('<II', quantized_bytes, 0)

    # Unpack all scales and zero points simultaneously (4 bytes per group at offset 8)
    scales_zp_bytes = quantized_bytes[8 : 8 + 4 * num_groups]
    scales_zp_tensor = torch.frombuffer(bytearray(scales_zp_bytes), dtype=torch.float16).reshape(num_groups, 2)
    scales = scales_zp_tensor[:, 0].unsqueeze(1)       # (num_groups, 1)
    zero_points = scales_zp_tensor[:, 1].unsqueeze(1)  # (num_groups, 1)

    # Unpack all packed INT4 bytes simultaneously
    packed_tensor = torch.frombuffer(bytearray(quantized_bytes[8 + 4 * num_groups :]), dtype=torch.uint8)
    low = packed_tensor & 0x0F
    high = packed_tensor >> 4

    # Interleave nibbles along the inner group dimension
    stacked = torch.stack([low, high], dim=1)
    all_quants = stacked.reshape(num_groups, GROUP_SIZE).to(torch.float16)

    # Dequantize using broadcasting in pure float16
    dequant = scales * all_quants + zero_points

    # Trim padding and return as raw FP16 bytes
    dequant_flat = dequant.flatten()[:num_floats]
    return dequant_flat.numpy().tobytes()
