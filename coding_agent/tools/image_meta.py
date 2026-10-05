"""Image metadata tool.

Extracts metadata (dimensions, format, colour mode, DPI, EXIF, file size)
from image files without loading pixel data into the LLM context.

Supported sources:
  - Regular files on disk (PNG, JPEG, BMP, GIF, TIFF, WEBP, …)
  - Images inside zip / SRDP packages (two levels deep: zip → nested-zip → image)
  - SVG files (parsed as XML — no PIL needed)

NOT included: pixel data / base64 encoding.
Use run_shell + Pillow if you need a histogram or colour analysis.
"""
from __future__ import annotations

import io
import re
import struct
import zipfile
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _cap, _cfg, _root, _within
from .srdp import _read_capped

# ---------------------------------------------------------------------------
# format detection from magic bytes (no PIL dependency for detection)
# ---------------------------------------------------------------------------

_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff",       "JPEG"),
    (b"GIF87a",             "GIF"),
    (b"GIF89a",             "GIF"),
    (b"BM",                 "BMP"),
    (b"II*\x00",            "TIFF"),  # little-endian
    (b"MM\x00*",            "TIFF"),  # big-endian
    (b"RIFF",               "WEBP"),  # + b"WEBP" at offset 8
    (b"\x00\x00\x01\x00",  "ICO"),
]


def _detect_format(data: bytes) -> str:
    for magic, fmt in _MAGIC:
        if data[:len(magic)] == magic:
            if fmt == "WEBP" and data[8:12] != b"WEBP":
                continue
            return fmt
    if data[:100].lstrip(b"\xef\xbb\xbf").lstrip().startswith((b"<svg", b"<?xml")):
        return "SVG"
    if data[:2] in (b"\xd0\xcf",):
        return "EMF/WMF"   # OLE compound (Windows metafile)
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# per-format metadata extractors (no PIL)
# ---------------------------------------------------------------------------

def _meta_png(data: bytes) -> dict:
    """Parse PNG IHDR chunk for dimensions and bit depth."""
    # PNG signature = 8 bytes, then chunks: length(4) + type(4) + data + crc(4)
    if len(data) < 24:
        return {}
    # IHDR is always first chunk, data starts at offset 16
    w = struct.unpack(">I", data[16:20])[0]
    h = struct.unpack(">I", data[20:24])[0]
    bit_depth = data[24]
    color_type = data[25]
    color_map = {0: "Grayscale", 2: "RGB", 3: "Indexed", 4: "Grayscale+Alpha", 6: "RGBA"}
    mode = color_map.get(color_type, f"type={color_type}")

    # pHYs chunk → DPI
    dpi = None
    idx = 8
    while idx + 12 < len(data):
        length = struct.unpack(">I", data[idx:idx+4])[0]
        chunk_type = data[idx+4:idx+8]
        if chunk_type == b"pHYs" and length == 9:
            px = struct.unpack(">I", data[idx+8:idx+12])[0]
            py = struct.unpack(">I", data[idx+12:idx+16])[0]
            unit = data[idx+16]
            if unit == 1:  # metres
                dpi = (round(px * 0.0254, 1), round(py * 0.0254, 1))
            break
        if chunk_type == b"IDAT":
            break
        idx += 12 + length

    result = {"width": w, "height": h, "mode": mode, "bit_depth": bit_depth}
    if dpi:
        result["dpi"] = f"{dpi[0]}x{dpi[1]}"
    return result


def _meta_jpeg(data: bytes) -> dict:
    """Parse JPEG SOF marker for dimensions; EXIF APP1 for DPI and camera info."""
    result: dict = {}
    i = 2  # skip SOI marker
    while i < len(data) - 4:
        if data[i] != 0xFF:
            break
        marker = data[i+1]
        length = struct.unpack(">H", data[i+2:i+4])[0]

        # SOF markers: C0-C3, C5-C7, C9-CB, CD-CF
        if marker in range(0xC0, 0xD0) and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 < len(data):
                result["bit_depth"] = data[i+4]
                result["height"] = struct.unpack(">H", data[i+5:i+7])[0]
                result["width"]  = struct.unpack(">H", data[i+7:i+9])[0]
                result["components"] = data[i+9]
                result["mode"] = {1: "Grayscale", 3: "YCbCr/RGB", 4: "CMYK"}.get(
                    data[i+9], f"comp={data[i+9]}")

        # APP1 = EXIF
        elif marker == 0xE1:
            app1 = data[i+4:i+2+length]
            if app1[:6] == b"Exif\x00\x00":
                _parse_exif_dpi(app1[6:], result)

        # APP0 = JFIF → DPI
        elif marker == 0xE0:
            if data[i+4:i+9] == b"JFIF\x00" and length >= 14:
                unit = data[i+11]
                xd = struct.unpack(">H", data[i+12:i+14])[0]
                yd = struct.unpack(">H", data[i+14:i+16])[0]
                if unit == 1 and xd:
                    result["dpi"] = f"{xd}x{yd}"

        if marker == 0xDA:  # SOS — compressed data starts, stop scanning
            break
        i += 2 + length

    return result


def _parse_exif_dpi(tiff_data: bytes, result: dict) -> None:
    """Extract XResolution, YResolution, Make, Model from TIFF/EXIF block."""
    try:
        if len(tiff_data) < 8:
            return
        endian = "<" if tiff_data[:2] == b"II" else ">"
        ifd_offset = struct.unpack(endian + "I", tiff_data[4:8])[0]
        n_entries = struct.unpack(endian + "H", tiff_data[ifd_offset:ifd_offset+2])[0]
        tags: dict[int, object] = {}
        for j in range(n_entries):
            base = ifd_offset + 2 + j * 12
            if base + 12 > len(tiff_data):
                break
            tag  = struct.unpack(endian + "H", tiff_data[base:base+2])[0]
            typ  = struct.unpack(endian + "H", tiff_data[base+2:base+4])[0]
            cnt  = struct.unpack(endian + "I", tiff_data[base+4:base+8])[0]
            val_raw = tiff_data[base+8:base+12]
            offset = struct.unpack(endian + "I", val_raw)[0]

            # RATIONAL (type 5): numerator/denominator at offset
            if typ == 5 and offset + 8 <= len(tiff_data):
                num = struct.unpack(endian + "I", tiff_data[offset:offset+4])[0]
                den = struct.unpack(endian + "I", tiff_data[offset+4:offset+8])[0]
                tags[tag] = num / den if den else 0
            # ASCII string
            elif typ == 2:
                str_offset = offset if cnt > 4 else base + 8
                if str_offset + cnt <= len(tiff_data):
                    tags[tag] = tiff_data[str_offset:str_offset+cnt].rstrip(b"\x00").decode("latin-1", errors="replace")
            # SHORT
            elif typ == 3:
                tags[tag] = struct.unpack(endian + "H", val_raw[:2])[0]

        xres = tags.get(0x011A)  # XResolution
        yres = tags.get(0x011B)  # YResolution
        unit = tags.get(0x0128, 2)  # ResolutionUnit (2=inch)
        if xres and unit == 2:
            result["dpi"] = f"{round(xres)}x{round(yres or xres)}"

        for tag_id, key in [(0x010F, "make"), (0x0110, "model"), (0x0132, "datetime")]:
            if tag_id in tags:
                result[key] = str(tags[tag_id])[:60]
    except Exception:
        pass  # EXIF parsing is best-effort


def _meta_svg(data: bytes) -> dict:
    """Extract width/height/viewBox from SVG XML."""
    try:
        text = data.decode("utf-8", errors="replace")
        result: dict = {"format": "SVG (vector, scalable)"}
        for attr in ("width", "height", "viewBox"):
            m = re.search(rf'{attr}=["\']([^"\']+)["\']', text)
            if m:
                result[attr] = m.group(1)
        # Count elements as complexity indicator
        n_elements = text.count("<")
        result["xml_elements"] = n_elements
        return result
    except Exception:
        return {}


def _meta_gif(data: bytes) -> dict:
    if len(data) < 10:
        return {}
    w = struct.unpack("<H", data[6:8])[0]
    h = struct.unpack("<H", data[8:10])[0]
    flags = data[10]
    n_colors = 2 ** ((flags & 0x07) + 1) if flags & 0x80 else 0
    return {"width": w, "height": h, "mode": "Indexed", "palette_colors": n_colors}


def _meta_bmp(data: bytes) -> dict:
    if len(data) < 26:
        return {}
    w = struct.unpack("<I", data[18:22])[0]
    h = abs(struct.unpack("<i", data[22:26])[0])
    bpp = struct.unpack("<H", data[28:30])[0]
    return {"width": w, "height": h, "bit_depth": bpp}


# ---------------------------------------------------------------------------
# PIL fallback for formats we don't parse manually (TIFF, WEBP, etc.)
# ---------------------------------------------------------------------------

def _meta_via_pil(data: bytes) -> dict:
    try:
        from PIL import Image, ExifTags
        img = Image.open(io.BytesIO(data))
        result: dict = {
            "width": img.size[0],
            "height": img.size[1],
            "mode": img.mode,
        }
        dpi = img.info.get("dpi")
        if dpi:
            result["dpi"] = f"{round(dpi[0])}x{round(dpi[1])}"
        try:
            exif = img.getexif()
            for tag_id, val in list(exif.items())[:6]:
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag in ("Make", "Model", "DateTime", "ImageDescription"):
                    result[tag.lower()] = str(val)[:80]
        except Exception:
            pass
        return result
    except Exception as e:
        return {"pil_error": str(e)}


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def _extract_metadata(data: bytes, filename: str) -> dict:
    fmt = _detect_format(data)
    base: dict = {
        "filename": Path(filename).name,
        "format": fmt,
        "file_size_bytes": len(data),
    }
    ext = Path(filename).suffix.lower()

    if fmt == "PNG":
        base.update(_meta_png(data))
    elif fmt == "JPEG":
        base.update(_meta_jpeg(data))
    elif fmt == "GIF":
        base.update(_meta_gif(data))
    elif fmt == "BMP":
        base.update(_meta_bmp(data))
    elif fmt == "SVG":
        base.update(_meta_svg(data))
    elif fmt in ("TIFF", "WEBP", "ICO") or ext in (".tiff", ".tif", ".webp", ".ico"):
        base.update(_meta_via_pil(data))
    else:
        base["note"] = "Format not recognised — no metadata extracted"

    return base


def _format_meta(meta: dict) -> str:
    lines = []
    order = [
        "filename", "format", "file_size_bytes",
        "width", "height", "mode", "bit_depth", "dpi",
        "palette_colors", "components",
        "make", "model", "datetime",
        "viewBox", "xml_elements",
        "note", "pil_error",
    ]
    shown = set()
    for key in order:
        if key in meta:
            val = meta[key]
            if key == "file_size_bytes":
                lines.append(f"  file_size:  {val:,} bytes  ({val/1024:.1f} KB)")
            elif key in ("width", "height"):
                # show together
                if "width" in meta and "height" in meta and "dimensions" not in shown:
                    lines.append(f"  dimensions: {meta['width']} x {meta['height']} px")
                    shown.add("dimensions")
                    shown.add("width"); shown.add("height")
            elif key not in shown:
                lines.append(f"  {key:<12}{val}")
            shown.add(key)
    # any remaining keys
    for key, val in meta.items():
        if key not in shown:
            lines.append(f"  {key:<12}{val}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# source loaders
# ---------------------------------------------------------------------------

def _load_from_zip(
    root: Path,
    zip_path: str,
    entry: str,
    inner_entry: str,
) -> tuple[bytes, str] | str:
    """Load image bytes from inside a zip (optionally two levels deep).
    Returns (bytes, name) or an error string.
    """
    p = (root / zip_path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {zip_path}"
    if not p.is_file():
        return f"Error: zip not found: {zip_path}"

    try:
        with zipfile.ZipFile(str(p)) as zf:
            if entry not in zf.namelist():
                return f"Error: '{entry}' not found in {zip_path}"
            raw = _read_capped(zf, entry)

            if not inner_entry:
                return raw, entry

            # Two-level: entry is itself a zip (bin/pptx/docx)
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                    if inner_entry not in inner.namelist():
                        close = [n for n in inner.namelist() if inner_entry.lower() in n.lower()]
                        hint = f"  Closest: {close[:3]}" if close else ""
                        return f"Error: '{inner_entry}' not found inside {entry}.{hint}"
                    return _read_capped(inner, inner_entry), inner_entry
            except zipfile.BadZipFile:
                return f"Error: '{entry}' is not a zip — cannot use inner_entry"

    except zipfile.BadZipFile:
        return f"Error: {zip_path} is not a valid zip"
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------

@tool
def read_image_meta(
    path: str,
    zip_entry: str = "",
    inner_entry: str = "",
    config: RunnableConfig = None,
) -> str:
    """Extract metadata from an image file without reading pixel data.

    Returns: filename, format, dimensions (WxH), colour mode, DPI,
    file size, and EXIF fields (make/model/datetime) when available.
    SVG files return width/height/viewBox from the XML.

    Three usage patterns:

    1. Regular file on disk:
       path="assets/logo.png"

    2. Image inside a zip / SRDP package:
       path="MainLiveDoc.pptx"   zip_entry="ppt/media/image1.png"
       (path is the outer zip, zip_entry is the image inside it)

    3. Image inside a nested zip (e.g. ExtContent .bin inside SRDP):
       path="package.zip"   zip_entry="ExtContent/abc.bin"
       inner_entry="ppt/media/image1.png"

    Supported formats: PNG, JPEG, GIF, BMP, SVG, TIFF, WEBP.
    Does NOT return pixel data or base64 — use run_shell + Pillow for that.
    """
    root = _root(config)

    # --- Case 1: image inside a zip (or nested zip) ---
    if zip_entry:
        result = _load_from_zip(root, path, zip_entry, inner_entry)
        if isinstance(result, str):
            return result   # error message
        data, name = result
        meta = _extract_metadata(data, name)
        header = f"Image: {path} / {zip_entry}" + (f" / {inner_entry}" if inner_entry else "")
        return header + "\n" + _format_meta(meta)

    # --- Case 2: regular file on disk ---
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    if not p.is_file():
        return f"Error: file not found: {path}"

    ext = p.suffix.lower()
    if ext not in {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg",
        ".tiff", ".tif", ".webp", ".ico", ".emf", ".wmf",
    }:
        return (
            f"Error: '{path}' does not look like an image file "
            f"(extension: {ext or 'none'}).\n"
            "For images inside a zip/SRDP, use the zip_entry parameter."
        )

    data = p.read_bytes()
    meta = _extract_metadata(data, p.name)
    return f"Image: {path}\n" + _format_meta(meta)
