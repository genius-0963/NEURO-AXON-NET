"""
Raw Microscopy Image Ingestion and Validation Engine.

Implements:
- Format allowlist (TIFF, PNG, JPEG) and bit-depth preservation
- Pre-decode SHA-256 calculation
- Corrupt and zero-sized image rejection
- Perceptual difference hashing (dHash) for duplicate detection
- Provenance metadata capture without fabrication
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute pre-decode SHA-256 hash of raw file bytes."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Source file does not exist: {file_path}")
    if p.stat().st_size == 0:
        raise ValueError(f"Zero-byte empty file rejected: {file_path}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_array: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute perceptual difference hash (dHash) for near-duplicate detection.
    """
    # Resize to (hash_size + 1, hash_size)
    pil_img = Image.fromarray(image_array.astype(np.uint8) if image_array.dtype != np.uint8 else image_array)
    resized = pil_img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32)
    diff = arr[:, 1:] > arr[:, :-1]
    # Convert bool array to hex string
    bit_str = "".join(["1" if b else "0" for b in diff.flatten()])
    return f"{int(bit_str, 2):0{hash_size * hash_size // 4}x}"


def read_microscopy_raw(
    file_path: Path | str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ingest a raw microscopy image preserving bit depth and metadata.

    Returns:
        image_data: np.ndarray of shape (H, W) or (H, W, C)
        metadata: Dict with file metadata, sha256, bit_depth, format, etc.
    """
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Disallowed image format '{ext}'. Must be one of {ALLOWED_EXTENSIONS}")

    sha256 = compute_file_sha256(p)

    metadata: Dict[str, Any] = {
        "file_path": str(p.resolve()),
        "file_name": p.name,
        "source_sha256": sha256,
        "file_size_bytes": p.stat().st_size,
        "format": ext.lstrip("."),
    }

    # TIFF reader with 16-bit support
    if ext in {".tif", ".tiff"} and HAS_TIFFFILE:
        try:
            arr = tifffile.imread(str(p))
            metadata["backend"] = "tifffile"
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable TIFF file {p}: {e}")
    else:
        try:
            with Image.open(p) as pil_img:
                pil_img.verify()
            with Image.open(p) as pil_img:
                arr = np.array(pil_img)
                metadata["backend"] = "pillow"
        except Exception as e:
            raise ValueError(f"Corrupt or invalid image file {p}: {e}")

    # Inspect dimensions and channels
    if arr.ndim == 2:
        h, w = arr.shape
        c = 1
    elif arr.ndim == 3:
        h, w, c = arr.shape
        if c not in (1, 3, 4):
            raise ValueError(f"Unsupported channel count {c} in {p}")
    else:
        raise ValueError(f"Unsupported array dimensionality {arr.ndim} in {p}")

    metadata["width"] = int(w)
    metadata["height"] = int(h)
    metadata["channels"] = int(c)
    metadata["dtype"] = str(arr.dtype)
    metadata["bit_depth"] = 16 if arr.dtype in (np.uint16, np.int16) else 8

    # Perceptual hash for near-duplicate auditing
    mono = arr if arr.ndim == 2 else arr[..., 0]
    metadata["dhash"] = compute_dhash(mono)

    return arr, metadata


def convert_to_model_channels(
    raw_array: np.ndarray,
    conversion_mode: str = "luminance_rec601",
) -> np.ndarray:
    """
    Convert raw image to single-channel 2D float32 representation for model ingestion.
    Preserves raw data without throwing away original multi-channel file.
    """
    if raw_array.ndim == 2:
        return raw_array.astype(np.float32)

    # Multi-channel
    c = raw_array.shape[2]
    if c == 1:
        return raw_array[:, :, 0].astype(np.float32)

    # RGB or RGBA: Rec. 601 Luma transform: Y = 0.299 R + 0.587 G + 0.114 B
    r = raw_array[:, :, 0].astype(np.float32)
    g = raw_array[:, :, 1].astype(np.float32)
    b = raw_array[:, :, 2].astype(np.float32)

    if conversion_mode == "luminance_rec601":
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        return luma
    elif conversion_mode == "green_channel":
        # Often highest contrast in PPD staining
        return g
    else:
        raise ValueError(f"Unknown channel conversion mode: {conversion_mode}")
