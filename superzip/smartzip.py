#!/usr/bin/env python3
"""
smartzip.py - Super Archive tool with shared dictionary + lossless media recompression.
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
Ibrahim Malek created the solution to compress model files using a shared dictionary 
trained via Zstandard, combined with vectorized group-wise INT4 quantization.
Usage:
  python smartzip.py create <folder> [archive.sa]
  python smartzip.py add <file> <archive.sa>
  python smartzip.py extract <archive.sa> [output_folder]
"""

import sys
import os
import struct
import subprocess
import tempfile
import zstandard as zstd

# -------------------- Optional Pillow for BMP support ----------------
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ----------------- External optimiser commands -----------------------
OPTIMISERS = {
    'png':  ['oxipng', '--opt', 'max', '--strip', 'all', '{file}'],
    'jpg':  ['jpegoptim', '--strip-all', '{file}'],
    'jpeg': ['jpegoptim', '--strip-all', '{file}'],
}

# ----------------- Magic byte detection ------------------------------
def detect_type_by_magic(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(12)
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if header.startswith(b'\xff\xd8\xff'):
        return 'jpg'
    if header.startswith(b'BM'):
        return 'bmp'
    if header.startswith(b'GIF8'):
        return 'gif'
    if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        return 'webp'
    return None

# ----------------- Optimisation helper --------------------------------
def run_optimiser(filepath):
    # Determine type by magic, fall back to extension
    magic_type = detect_type_by_magic(filepath)
    if magic_type:
        file_ext = magic_type
    else:
        file_ext = os.path.splitext(filepath)[1].lstrip('.')
    lower = file_ext.lower()
    if lower not in OPTIMISERS:
        # No optimiser – return original bytes
        with open(filepath, 'rb') as f:
            return f.read()

    # Special case: BMP -> PNG conversion + optimisation
    if lower == 'bmp':
        if not HAS_PIL:
            return open(filepath, 'rb').read()
        try:
            img = Image.open(filepath)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                png_path = tmp.name
            img.save(png_path, 'PNG', optimize=True)
            result = run_optimiser(png_path)
            os.unlink(png_path)
            return result
        except Exception:
            return open(filepath, 'rb').read()

    # For PNG / JPEG: copy to a temp file, run external optimiser
    with tempfile.NamedTemporaryFile(suffix='.'+lower, delete=False) as tmp:
        tmp.write(open(filepath, 'rb').read())
        tmp_path = tmp.name

    cmd = [arg.replace('{file}', tmp_path) for arg in OPTIMISERS[lower]]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(tmp_path, 'rb') as f:
            result = f.read()
        os.unlink(tmp_path)
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        os.unlink(tmp_path)
        with open(filepath, 'rb') as f:
            return f.read()

# ----------------- FP32 Tensor pre-processing (Bit-plane separation) ---
def preprocess_fp32_tensor(raw_bytes):
    """Reorganise FP32 bytes to group sign, exponent, mantissa separately.
    Input: raw bytes of a float32 array (length must be multiple of 4).
    Returns: bytes that are the concatenation of all signs (packed bits), 
             all exponents, and all mantissa bytes.
    """
    num_floats = len(raw_bytes) // 4
    # Unpack as list of 32-bit ints (little-endian, same as numpy default)
    ints = struct.unpack('<' + 'I'*num_floats, raw_bytes)
    
    signs = []
    exponents = []
    mantissas = []
    for val in ints:
        sign = (val >> 31) & 1
        exp  = (val >> 23) & 0xFF
        mant = val & 0x7FFFFF  # 23 bits
        signs.append(sign)
        exponents.append(exp)
        # Convert mantissa to 3 bytes (big-endian for easier grouping)
        mantissas.append((mant >> 16) & 0xFF)
        mantissas.append((mant >> 8) & 0xFF)
        mantissas.append(mant & 0xFF)
    
    # Pack signs into bytes (8 signs per byte)
    sign_bytes = []
    for i in range(0, num_floats, 8):
        batch = signs[i:i+8]
        byte_val = 0
        for j, s in enumerate(batch):
            if s:
                byte_val |= (1 << j)
        sign_bytes.append(byte_val)
    
    # Combine: first a header with num_floats (4 bytes), then sign bytes, then exponent bytes, then mantissa bytes
    header = struct.pack('<I', num_floats)
    return header + bytes(sign_bytes) + bytes(exponents) + bytes(mantissas)

def unprocess_fp32_tensor(processed_bytes):
    """Reverse the preprocess_fp32_tensor transformation.
    Returns original raw float32 bytes.
    """
    num_floats = struct.unpack('<I', processed_bytes[:4])[0]
    pos = 4
    
    sign_bytes_len = (num_floats + 7) // 8
    sign_bytes = processed_bytes[pos:pos+sign_bytes_len]
    pos += sign_bytes_len
    
    exponent_bytes = processed_bytes[pos:pos+num_floats]
    pos += num_floats
    
    mantissa_bytes = processed_bytes[pos:pos+num_floats*3]
    # Now rebuild floats
    ints = []
    for i in range(num_floats):
        sign_byte = sign_bytes[i // 8]
        sign = (sign_byte >> (i % 8)) & 1
        exp = exponent_bytes[i]
        m0 = mantissa_bytes[i*3]
        m1 = mantissa_bytes[i*3+1]
        m2 = mantissa_bytes[i*3+2]
        mant = (m0 << 16) | (m1 << 8) | m2
        val = (sign << 31) | (exp << 23) | mant
        ints.append(val)
    return struct.pack('<' + 'I'*num_floats, *ints)

def fp32_to_fp16(raw_bytes):
    """Convert raw bytes of float32 array to float16 (half precision) bytes."""
    if len(raw_bytes) % 4 != 0:
        raise ValueError("Invalid float32 buffer length")
    try:
        import numpy as np
        arr = np.frombuffer(raw_bytes, dtype=np.float32)
        return arr.astype(np.float16).tobytes()
    except ImportError:
        num = len(raw_bytes) // 4
        fp32 = struct.unpack('<' + 'f'*num, raw_bytes)
        fp16 = []
        for val in fp32:
            fp16.append(struct.pack('<e', val))
        return b''.join(fp16)

def fp16_to_fp32(fp16_bytes):
    """Convert float16 bytes back to float32 bytes."""
    try:
        import numpy as np
        arr = np.frombuffer(fp16_bytes, dtype=np.float16)
        return arr.astype(np.float32).tobytes()
    except ImportError:
        num = len(fp16_bytes) // 2
        fp32 = []
        for i in range(num):
            half = fp16_bytes[i*2 : i*2+2]
            val = struct.unpack('<e', half)[0]
            fp32.append(struct.pack('<f', val))
        return b''.join(fp32)

# ----------------- Dictionary training --------------------------------

def train_dictionary(samples, max_size=128*1024):
    """samples: list of bytes objects."""
    if not samples:
        return b''
    total = sum(len(s) for s in samples)
    if total < 512:
        return b''
    dict_size = max(256, min(max_size, total))
    try:
        dict_obj = zstd.train_dictionary(dict_size, samples)
        return dict_obj.as_bytes()
    except Exception:
        return b''

# ----------------- Compression / Decompression ------------------------
def compress_with_dict(data, dict_bytes, level=3):
    if isinstance(dict_bytes, bytes) and len(dict_bytes) > 0:
        zdict = zstd.ZstdCompressionDict(dict_bytes)
    else:
        zdict = None
    cctx = zstd.ZstdCompressor(level=level, dict_data=zdict)
    return cctx.compress(data)

def decompress_with_dict(compressed, dict_bytes):
    if isinstance(dict_bytes, bytes) and len(dict_bytes) > 0:
        zdict = zstd.ZstdCompressionDict(dict_bytes)
    else:
        zdict = None
    dctx = zstd.ZstdDecompressor(dict_data=zdict)
    return dctx.decompress(compressed)

# ----------------- Binary container format ----------------------------
MAGIC = b'SZIP'
VERSION = 1

def _pack_entry(filename, uncomp_len, comp_data):
    fname = filename.encode('utf-8')
    return struct.pack('<H', len(fname)) + fname + struct.pack('<II', uncomp_len, len(comp_data)) + comp_data

def _write_sa(filepath, dict_bytes, file_list):
    """file_list: [(filename, uncompressed_len, compressed_data), ...]"""
    with open(filepath, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', len(file_list)))
        for fname, unc_len, comp in file_list:
            f.write(_pack_entry(fname, unc_len, comp))
        f.write(struct.pack('<I', len(dict_bytes)))
        f.write(dict_bytes)

def _read_sa(filepath):
    with open(filepath, 'rb') as f:
        if f.read(4) != MAGIC:
            raise ValueError('Not a valid SuperZIP archive')
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

# ----------------- Create archive -------------------------------------
def create_archive(input_folder, output_path):
    folder = os.path.abspath(input_folder)
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not all_files:
        print("No files found in the folder.")
        return

    # 1. Optimise
    print("Optimising files (lossless/quantized) ...")
    opt_map = {}   # path -> bytes
    is_int4_map = {} # path -> bool
    for full_path in all_files:
        base = os.path.basename(full_path)
        is_int4 = False
        # If it's a .bin file and not an image, apply tensor pre-processing
        if base.lower().endswith('.bin') and not detect_type_by_magic(full_path):
            with open(full_path, 'rb') as f:
                raw = f.read()
            if len(raw) % 4 == 0 and len(raw) >= 4:
                try:
                    try:
                        from .quantization import quantize_fp32_to_int4_groupwise as quantize_weights
                    except ImportError:
                        from quantization import quantize_fp32_to_int4_groupwise as quantize_weights
                    
                    opt_data = quantize_weights(raw)
                    is_int4 = True
                    print(f"  {base}: FP32->INT4 Group-wise ({len(raw)} -> {len(opt_data)} bytes)")
                except Exception as e:
                    print(f"  Warning: INT4 group-wise quantization failed for {base}, using raw ({e})")
                    opt_data = raw
            else:
                opt_data = raw
        else:
            opt_data = run_optimiser(full_path)
            print(f"  {base}: {os.path.getsize(full_path)} -> {len(opt_data)} bytes")
        opt_map[full_path] = opt_data
        is_int4_map[full_path] = is_int4

    # 2. Train dictionary
    print("\nTraining shared dictionary ...")
    samples = list(opt_map.values())
    dict_bytes = train_dictionary(samples)

    # 3. Compress
    print("\nCompressing ...")
    file_info = []
    for full_path in all_files:
        base = os.path.basename(full_path)
        uncomp = opt_map[full_path]
        comp = compress_with_dict(uncomp, dict_bytes)
        fname = base + '.int4g' if is_int4_map[full_path] else base
        file_info.append((fname, len(uncomp), comp))
        print(f"  {fname}: {len(uncomp)} -> {len(comp)} bytes")

    # 4. Write container
    _write_sa(output_path, dict_bytes, file_info)
    print(f"\n[SUCCESS] Archive created: {output_path}")

# ----------------- Add to archive -------------------------------------
def add_to_archive(archive_path, new_file_path):
    dict_bytes, old_files = _read_sa(archive_path)

    base = os.path.basename(new_file_path)
    is_fp16 = False
    if base.lower().endswith('.bin') and not detect_type_by_magic(new_file_path):
        with open(new_file_path, 'rb') as f:
            raw = f.read()
        if len(raw) % 4 == 0 and len(raw) >= 4:
            try:
                new_uncomp = fp32_to_fp16(raw)
                is_fp16 = True
                print(f"  {base}: FP32->FP16 ({len(raw)} -> {len(new_uncomp)} bytes)")
            except Exception:
                new_uncomp = raw
        else:
            new_uncomp = raw
    else:
        new_uncomp = run_optimiser(new_file_path)

    new_comp = compress_with_dict(new_uncomp, dict_bytes)
    fname = base + '.fp16' if is_fp16 else base

    new_files = old_files + [(fname, len(new_uncomp), new_comp)]
    _write_sa(archive_path, dict_bytes, new_files)
    print(f"[SUCCESS] Added '{new_file_path}' to archive. Now contains {len(new_files)} file(s).")

# ----------------- Extract archive ------------------------------------
def extract_archive(archive_path, output_folder):
    print(f"Reading archive: {archive_path}")
    dict_bytes, file_list = _read_sa(archive_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for fname, unc_len, comp_data in file_list:
        is_fp16 = False
        if fname.endswith('.fp16'):
            is_fp16 = True
            out_name = fname[:-5]
        else:
            out_name = fname
            
        print(f"Extracting: {out_name} (from {'FP16' if is_fp16 else 'original'} data)")
        data = decompress_with_dict(comp_data, dict_bytes)
        if is_fp16:
            data = fp16_to_fp32(data)
            
        out_path = os.path.join(output_folder, out_name)
        with open(out_path, 'wb') as f:
            f.write(data)
    print(f"[SUCCESS] Extracted {len(file_list)} file(s) to '{output_folder}'")

# ----------------- CLI -------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python smartzip.py create <folder> [archive.sa]")
        print("  python smartzip.py add <file> <archive.sa>")
        print("  python smartzip.py extract <archive.sa> [output_folder]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "create":
        folder = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else "archive.sa"
        create_archive(folder, out)
    elif cmd == "add":
        filepath = sys.argv[2]
        archive = sys.argv[3]
        add_to_archive(archive, filepath)
    elif cmd == "extract":
        archive = sys.argv[2]
        out_folder = sys.argv[3] if len(sys.argv) > 3 else "extracted"
        extract_archive(archive, out_folder)
    else:
        print("Unknown command.")

if __name__ == "__main__":
    main()
