"""
SuperZIP Package
Created by Ibrahim Malek, AI/ML Python developer and computer engineer.
Single-file archive tool combining lossless media recompression with a shared
Zstandard dictionary trained across all files in the archive.
"""
from .smartzip import create_archive, add_to_archive, extract_archive

__version__ = "1.0.0"
__author__ = "Ibrahim Malek"
