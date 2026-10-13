#!/usr/bin/env python3
"""
smartzip.py - Super Archive tool with shared dictionary + lossless media recompression.
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
Ibrahim Malek created the solution to compress a folder of files into a single archive
using a shared dictionary trained via Zstandard.
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
    print("Optimising files (lossless) ...")
    opt_map = {}   # path -> bytes
    for full_path in all_files:
        base = os.path.basename(full_path)
        opt_data = run_optimiser(full_path)
        print(f"  {base}: {os.path.getsize(full_path)} -> {len(opt_data)} bytes")
        opt_map[full_path] = opt_data

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
        file_info.append((base, len(uncomp), comp))
        print(f"  {base}: {len(uncomp)} -> {len(comp)} bytes")

    # 4. Write container
    _write_sa(output_path, dict_bytes, file_info)
    print(f"\n[SUCCESS] Archive created: {output_path}")

# ----------------- Add to archive -------------------------------------
def add_to_archive(archive_path, new_file_path):
    dict_bytes, old_files = _read_sa(archive_path)

    base = os.path.basename(new_file_path)
    new_uncomp = run_optimiser(new_file_path)
    new_comp = compress_with_dict(new_uncomp, dict_bytes)

    new_files = old_files + [(base, len(new_uncomp), new_comp)]
    _write_sa(archive_path, dict_bytes, new_files)
    print(f"[SUCCESS] Added '{new_file_path}' to archive. Now contains {len(new_files)} file(s).")

# ----------------- Extract archive ------------------------------------
def extract_archive(archive_path, output_folder):
    print(f"Reading archive: {archive_path}")
    dict_bytes, file_list = _read_sa(archive_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for fname, unc_len, comp_data in file_list:
        print(f"Extracting: {fname}")
        data = decompress_with_dict(comp_data, dict_bytes)
        out_path = os.path.join(output_folder, fname)
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
