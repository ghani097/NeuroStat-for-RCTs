"""Dependency-free structural and pixel QA for BrainNet PNG output."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

from .artifacts import BrainNetError

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_distance:
        return left
    return above if above_distance <= upper_distance else upper_left


def inspect_png(
    path: str | Path,
    *,
    width_pixels: int,
    height_pixels: int,
    dpi: int,
    dpi_tolerance: float = 1.0,
) -> dict[str, Any]:
    """Require exact dimensions, sane DPI metadata, and nonconstant 8-bit pixels."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise BrainNetError(f"rendered PNG is missing or empty: {path}")
    content = path.read_bytes()
    if not content.startswith(_SIGNATURE):
        raise BrainNetError("rendered image is not a PNG")
    offset = len(_SIGNATURE)
    ihdr = None
    physical = None
    compressed = bytearray()
    saw_end = False
    while offset < len(content):
        if offset + 12 > len(content):
            raise BrainNetError("PNG chunk framing is truncated")
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        kind = content[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(content):
            raise BrainNetError("PNG chunk payload is truncated")
        payload = content[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", content[offset + 8 + length : end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise BrainNetError("PNG chunk CRC is invalid")
        if kind == b"IHDR":
            if ihdr is not None or len(payload) != 13:
                raise BrainNetError("PNG IHDR is invalid")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif kind == b"pHYs":
            if len(payload) != 9:
                raise BrainNetError("PNG pHYs metadata is invalid")
            physical = struct.unpack(">IIB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            saw_end = True
            break
        offset = end
    if ihdr is None or not saw_end or not compressed:
        raise BrainNetError("PNG lacks required IHDR, IDAT, or IEND chunks")
    width, height, depth, color_type, compression, filtering, interlace = ihdr
    if (width, height) != (width_pixels, height_pixels):
        raise BrainNetError(
            f"PNG dimensions differ: {(width, height)} != "
            f"{(width_pixels, height_pixels)}"
        )
    if (
        depth != 8
        or color_type not in _CHANNELS
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise BrainNetError(
            "PNG must be noninterlaced 8-bit grayscale/RGB/indexed/RGBA"
        )
    if physical is None or physical[2] != 1:
        raise BrainNetError("PNG must record physical resolution in metres")
    dpi_x = physical[0] * 0.0254
    dpi_y = physical[1] * 0.0254
    if abs(dpi_x - dpi) > dpi_tolerance or abs(dpi_y - dpi) > dpi_tolerance:
        raise BrainNetError(
            f"PNG DPI differs beyond tolerance: {(dpi_x, dpi_y)} != {dpi}"
        )
    channels = _CHANNELS[color_type]
    stride = width * channels
    try:
        raw = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise BrainNetError("PNG image payload cannot be decompressed") from exc
    if len(raw) != height * (stride + 1):
        raise BrainNetError("PNG decompressed payload has an unexpected size")
    previous = bytearray(stride)
    extrema = [[255, 0] for _ in range(channels)]
    sums = [0 for _ in range(channels)]
    sums_of_squares = [0 for _ in range(channels)]
    unique_pixels: set[bytes] = set()
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        encoded = raw[cursor : cursor + stride]
        cursor += stride
        if filter_type > 4:
            raise BrainNetError("PNG uses an unknown row filter")
        row = bytearray(stride)
        for index, encoded_value in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            predictor = (
                0
                if filter_type == 0
                else left
                if filter_type == 1
                else above
                if filter_type == 2
                else (left + above) // 2
                if filter_type == 3
                else _paeth(left, above, upper_left)
            )
            row[index] = (encoded_value + predictor) & 0xFF
        for index in range(0, stride, channels):
            pixel = bytes(row[index : index + channels])
            if len(unique_pixels) < 2:
                unique_pixels.add(pixel)
            for channel, value in enumerate(pixel):
                extrema[channel][0] = min(extrema[channel][0], value)
                extrema[channel][1] = max(extrema[channel][1], value)
                sums[channel] += value
                sums_of_squares[channel] += value * value
        previous = row
    pixels = width * height
    standard_deviations = [
        max(
            0.0,
            sums_of_squares[channel] / pixels - (sums[channel] / pixels) ** 2,
        )
        ** 0.5
        for channel in range(channels)
    ]
    display_channels = min(channels, 3)
    if (
        len(unique_pixels) < 2
        or max(
            extrema[channel][1] - extrema[channel][0]
            for channel in range(display_channels)
        )
        < 8
        or max(standard_deviations[:display_channels]) < 1.0
    ):
        raise BrainNetError("PNG is blank or constant")
    return {
        "status": "passed",
        "sha256_pending": True,
        "bytes": len(content),
        "width_pixels": width,
        "height_pixels": height,
        "bit_depth": depth,
        "color_type": color_type,
        "channels": channels,
        "channel_extrema": extrema,
        "channel_standard_deviation": standard_deviations,
        "minimum_display_channel_std": 1.0,
        "minimum_display_dynamic_range": 8,
        "dpi_x": dpi_x,
        "dpi_y": dpi_y,
        "dpi_requested": dpi,
        "dpi_tolerance": dpi_tolerance,
        "nonconstant": True,
    }
