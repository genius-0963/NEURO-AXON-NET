"""
Unit tests for raw microscopy ingestion and validation logic (Section 5.1).
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from neuroaxonnet.data.ingestion import (
    compute_file_sha256,
    compute_dhash,
    read_microscopy_raw,
    convert_to_model_channels,
)


def test_empty_file_rejected(tmp_path: Path):
    empty_file = tmp_path / "zero_bytes.png"
    empty_file.touch()
    with pytest.raises(ValueError, match="Zero-byte empty file"):
        compute_file_sha256(empty_file)


def test_disallowed_extension_rejected(tmp_path: Path):
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not an image")
    with pytest.raises(ValueError, match="Disallowed image format"):
        read_microscopy_raw(bad_file)


def test_corrupt_image_rejected(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\nCorruptPayloadGarbage")
    with pytest.raises(ValueError, match="Corrupt or invalid"):
        read_microscopy_raw(corrupt_file)


def test_valid_png_ingestion(tmp_path: Path):
    valid_file = tmp_path / "valid.png"
    arr = np.random.randint(0, 255, size=(187, 187), dtype=np.uint8)
    Image.fromarray(arr).save(valid_file)

    loaded, meta = read_microscopy_raw(valid_file)
    assert loaded.shape == (187, 187)
    assert meta["width"] == 187
    assert meta["height"] == 187
    assert meta["bit_depth"] == 8
    assert "source_sha256" in meta
    assert len(meta["source_sha256"]) == 64
    assert "dhash" in meta


def test_rgb_channel_conversion():
    rgb_arr = np.ones((100, 100, 3), dtype=np.uint8) * 100
    luma = convert_to_model_channels(rgb_arr, conversion_mode="luminance_rec601")
    assert luma.shape == (100, 100)
    assert np.isclose(luma[0, 0], 100.0)
