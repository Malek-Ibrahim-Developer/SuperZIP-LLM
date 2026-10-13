# SuperZIP: Single-File Archives with a Shared Zstandard Dictionary

**SuperZIP**, created by **Ibrahim Malek** (AI/ML Python developer and computer engineer), packs a folder of files into one `.sa` archive. Before compressing, it recompresses images losslessly. It then trains one Zstandard dictionary on all the files and compresses every file with it, so files that look alike compress better together.

---

## ✨ Features

- **Single `.sa` archive**: the whole folder goes into one file.
- **Shared Zstandard dictionary**: one dictionary (up to 128 KB), trained on every file in the archive, is used to compress each entry.
- **Lossless media recompression**: PNG files go through `oxipng` and JPEG files through `jpegoptim`, when those tools are installed. File types are detected by magic bytes, falling back to the file extension.
- **Append without rebuilding**: `add` compresses a new file with the archive's existing dictionary.

---

## 🚀 Quick Start

```bash
pip install .
```

```bash
# Create an archive from a folder (default output: archive.sa)
smartzip create my_folder my_folder.sa

# Add one more file to an existing archive
smartzip add notes.txt my_folder.sa

# Extract an archive (default output folder: extracted)
smartzip extract my_folder.sa restored/
```

From Python:

```python
from superzip import create_archive, add_to_archive, extract_archive

create_archive("my_folder", "my_folder.sa")
add_to_archive("my_folder.sa", "notes.txt")
extract_archive("my_folder.sa", "restored")
```

### Optional tools

| Tool | Used for |
| :--- | :--- |
| [`oxipng`](https://github.com/shssoichiro/oxipng) | Lossless PNG optimisation |
| [`jpegoptim`](https://github.com/tjko/jpegoptim) | Lossless JPEG optimisation |

If a tool is missing, SuperZIP stores the file's original bytes.

---

## 📦 `.sa` Archive Format

All integers are little-endian.

| Field | Type | Notes |
| :--- | :--- | :--- |
| Magic | 4 bytes | `SZIP` |
| Version | `uint32` | Currently `1` |
| Entry count | `uint32` | |
| Entries | repeated | See below |
| Dictionary length | `uint32` | `0` if no dictionary was trained |
| Dictionary | bytes | Zstandard dictionary |

Each entry:

| Field | Type |
| :--- | :--- |
| Name length | `uint16` |
| Name | UTF-8 bytes |
| Uncompressed length | `uint32` |
| Compressed length | `uint32` |
| Compressed data | bytes |

---

## ⚠️ Current Limitations

- Only files directly inside the folder are archived; subfolders are skipped.
- PNG and JPEG files are extracted in their optimised form: the image data is the same, but the bytes differ from the original and `--strip-all` removes metadata such as EXIF.
- The whole archive is read into memory on `add` and `extract`.
- Each entry is limited to 4 GB (`uint32` lengths).

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
