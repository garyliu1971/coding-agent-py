"""SRDP debug package tools.

SRDP (Seismic Reproduction Debug Package) is a zip archive containing:
  - index.xml                      : manifest (engine version, file map)
  - MainRequestDocumentInput.xml   : original generation request parameters
  - AuthToken.xml                  : tenant / server info
  - UserInfo.xml                   : customer user data
  - MainPresentationObjectInfo.xml : presentation object metadata
  - MainLiveDoc.pptx / .docx       : master template
  - InstanceLiveDoc.pptx / .docx   : generated output
  - SharedComponentObject/*.xml    : shared component metadata
  - ExtContent/*.xml               : external content metadata
  - ExtContent/*.bin               : nested zip (actual shared component file)
  - DataSourceBytes/*.bin          : data source binary (Excel/etc)
  - DataSourceSerivceContent/*.xml : data source service metadata

Two tools are provided:

  srdp_list   — list all entries in the SRDP with sizes and types
  srdp_read   — read a specific entry; handles nested zips inside .bin files
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _cfg, _root, _within

# Zip-bomb guard: no single entry may inflate beyond this many bytes in memory.
_MAX_ENTRY_BYTES = 256 * 1024 * 1024


class EntryTooLarge(ValueError):
    pass


def _read_capped(zf: zipfile.ZipFile, name, limit: int = _MAX_ENTRY_BYTES) -> bytes:
    """``zf.read(name)`` that refuses entries inflating past ``limit`` (declared sizes can lie)."""
    with zf.open(name) as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise EntryTooLarge(f"zip entry {getattr(name, 'filename', name)} is larger than {limit // (1024 * 1024)} MB uncompressed; refusing to read it")
    return data

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {
    ".xml", ".json", ".txt", ".csv", ".html", ".htm", ".md",
    ".cs", ".ts", ".js", ".py", ".yaml", ".yml", ".config", ".csproj",
}

_BINARY_ZIP_EXTENSIONS = {".bin", ".zip", ".pptx", ".docx", ".xlsx"}


def _is_text(name: str) -> bool:
    return Path(name).suffix.lower() in _TEXT_EXTENSIONS


def _resolve_srdp(root: Path, srdp_path: str) -> tuple[Path, str]:
    """Resolve srdp_path relative to root; return (abs_path, display_path)."""
    p = (root / srdp_path).resolve()
    return p, srdp_path


def _cap_text(text: str, limit: int = 20_000) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text)-limit:,} chars]"
    return text


def _read_entry_text(data: bytes, name: str, limit: int = 20_000) -> str:
    """Decode bytes as UTF-8 text."""
    text = data.decode("utf-8", errors="replace")
    return _cap_text(text, limit)


def _list_zip_entries(zf: zipfile.ZipFile, prefix: str = "") -> list[str]:
    lines = []
    for info in zf.infolist():
        size = info.file_size
        ext = Path(info.filename).suffix.lower()
        kind = "text" if _is_text(info.filename) else ("zip/bin" if ext in _BINARY_ZIP_EXTENSIONS else "binary")
        lines.append(f"  {prefix}{info.filename}  ({size:,} B)  [{kind}]")
    return lines


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@tool
def srdp_list(
    srdp_path: str,
    config: RunnableConfig = None,
) -> str:
    """List all files inside an SRDP package (or any zip file).

    Shows each entry with its size and type (text / zip-bin / binary).
    For .bin entries that are nested zips (SharedComponent / ExtContent),
    also lists their internal files.

    Args:
        srdp_path: Path to the .zip / .srdp file, relative to project root.
    """
    root = _root(config)
    p, display = _resolve_srdp(root, srdp_path)
    if not _within(root, p):
        return f"Error: path escapes project root: {srdp_path}"
    if not p.is_file():
        return f"Error: file not found: {srdp_path}"

    try:
        with zipfile.ZipFile(str(p)) as zf:
            lines = [f"SRDP: {display}  ({p.stat().st_size:,} bytes total)"]
            lines.append(f"  Entries: {len(zf.infolist())}")
            lines.append("")

            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                size = info.file_size
                ext = Path(info.filename).suffix.lower()
                kind = "text" if _is_text(info.filename) else (
                    "zip/bin" if ext in _BINARY_ZIP_EXTENSIONS else "binary"
                )
                lines.append(f"  {info.filename}  ({size:,} B)  [{kind}]")

                # If it's a .bin, try to peek inside as a nested zip
                if ext == ".bin" and size > 0:
                    try:
                        raw = _read_capped(zf, info.filename)
                        with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                            inner_entries = inner.infolist()
                            lines.append(f"    └─ nested zip ({len(inner_entries)} entries):")
                            for ie in inner_entries[:20]:
                                ikind = "text" if _is_text(ie.filename) else "binary"
                                lines.append(
                                    f"      {ie.filename}  ({ie.file_size:,} B)  [{ikind}]"
                                )
                            if len(inner_entries) > 20:
                                lines.append(f"      ... and {len(inner_entries)-20} more")
                    except Exception:
                        lines.append("    └─ (not a zip)")

            return "\n".join(lines)
    except zipfile.BadZipFile:
        return f"Error: {srdp_path} is not a valid zip file."
    except Exception as exc:
        return f"Error reading {srdp_path}: {exc}"


@tool
def srdp_read(
    srdp_path: str,
    entry: str,
    inner_entry: str = "",
    config: RunnableConfig = None,
) -> str:
    """Read a specific file from inside an SRDP package.

    Handles two levels:
    1. Top-level entry  (e.g. "index.xml", "UserInfo.xml")
    2. Nested entry inside a .bin (e.g. entry="ExtContent/abc.bin",
       inner_entry="ppt/slides/slide1.xml")

    For binary files that are not zips (images, fonts, etc.) returns a
    hex summary instead of raw bytes.

    Args:
        srdp_path:   Path to the SRDP zip, relative to project root.
        entry:       Entry name inside the SRDP (from srdp_list output).
        inner_entry: If entry is a .bin nested zip, the file inside it.
                     Leave empty to list the nested zip's contents instead.
    """
    root = _root(config)
    p, display = _resolve_srdp(root, srdp_path)
    if not _within(root, p):
        return f"Error: path escapes project root: {srdp_path}"
    if not p.is_file():
        return f"Error: file not found: {srdp_path}"

    cfg = _cfg(config)
    limit = int(cfg.get("file_read_limit", 20_000))

    try:
        with zipfile.ZipFile(str(p)) as zf:
            # Check entry exists
            names = zf.namelist()
            if entry not in names:
                close = [n for n in names if entry.lower() in n.lower()]
                hint = f"  Did you mean: {close[:5]}" if close else ""
                return f"Error: entry '{entry}' not found in {display}.{hint}\nUse srdp_list to see all entries."

            raw = _read_capped(zf, entry)
            ext = Path(entry).suffix.lower()

            # --- Case 1: top-level text file ---
            if _is_text(entry) and not inner_entry:
                return (
                    f"=== {display} / {entry} ({len(raw):,} bytes) ===\n"
                    + _read_entry_text(raw, entry, limit)
                )

            # --- Case 2: .bin that is a nested zip ---
            if ext == ".bin":
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                        if not inner_entry:
                            # List the nested zip contents
                            lines = [
                                f"=== {display} / {entry} — nested zip contents ===",
                                f"  Entries: {len(inner.infolist())}",
                            ]
                            for ie in inner.infolist():
                                ikind = "text" if _is_text(ie.filename) else "binary"
                                lines.append(
                                    f"  {ie.filename}  ({ie.file_size:,} B)  [{ikind}]"
                                )
                            lines.append(
                                "\nTo read a file inside: call srdp_read with "
                                f"entry='{entry}' and inner_entry='<filename>'"
                            )
                            return "\n".join(lines)

                        # Read the inner entry
                        inner_names = inner.namelist()
                        if inner_entry not in inner_names:
                            close = [n for n in inner_names if inner_entry.lower() in n.lower()]
                            hint = f"  Did you mean: {close[:5]}" if close else ""
                            return (
                                f"Error: inner entry '{inner_entry}' not found in {entry}.{hint}\n"
                                "Call srdp_read without inner_entry to list the nested zip."
                            )
                        inner_raw = _read_capped(inner, inner_entry)
                        if _is_text(inner_entry):
                            return (
                                f"=== {display} / {entry} / {inner_entry} "
                                f"({len(inner_raw):,} bytes) ===\n"
                                + _read_entry_text(inner_raw, inner_entry, limit)
                            )
                        else:
                            return (
                                f"=== {display} / {entry} / {inner_entry} "
                                f"({len(inner_raw):,} bytes) [binary] ===\n"
                                f"Binary file — not directly readable as text.\n"
                                f"First 64 bytes (hex): {inner_raw[:64].hex()}"
                            )
                except zipfile.BadZipFile:
                    # .bin but not a zip — show hex summary
                    return (
                        f"=== {display} / {entry} ({len(raw):,} bytes) [binary, not a zip] ===\n"
                        f"First 64 bytes (hex): {raw[:64].hex()}"
                    )

            # --- Case 3: other binary (pptx, docx, etc.) ---
            if ext in {".pptx", ".docx", ".xlsx"}:
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                        if not inner_entry:
                            lines = [
                                f"=== {display} / {entry} — Office zip contents ===",
                                f"  Entries: {len(inner.infolist())}",
                            ]
                            for ie in sorted(inner.infolist(), key=lambda x: x.filename):
                                ikind = "text" if _is_text(ie.filename) else "binary"
                                lines.append(
                                    f"  {ie.filename}  ({ie.file_size:,} B)  [{ikind}]"
                                )
                            lines.append(
                                "\nTo read a file inside: call srdp_read with "
                                f"entry='{entry}' and inner_entry='<filename>'"
                            )
                            return "\n".join(lines)

                        inner_names = inner.namelist()
                        if inner_entry not in inner_names:
                            close = [n for n in inner_names if inner_entry.lower() in n.lower()]
                            hint = f"  Did you mean: {close[:5]}" if close else ""
                            return (
                                f"Error: inner entry '{inner_entry}' not found.{hint}\n"
                                "Call without inner_entry to list contents."
                            )
                        inner_raw = _read_capped(inner, inner_entry)
                        if _is_text(inner_entry):
                            return (
                                f"=== {display} / {entry} / {inner_entry} "
                                f"({len(inner_raw):,} bytes) ===\n"
                                + _read_entry_text(inner_raw, inner_entry, limit)
                            )
                        else:
                            return (
                                f"[binary: {len(inner_raw):,} bytes, not text-readable]"
                            )
                except zipfile.BadZipFile:
                    pass

            return (
                f"=== {display} / {entry} ({len(raw):,} bytes) [binary] ===\n"
                f"Binary file — not directly readable as text.\n"
                f"First 64 bytes (hex): {raw[:64].hex()}"
            )

    except zipfile.BadZipFile:
        return f"Error: {srdp_path} is not a valid zip file."
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# srdp_grep
# ---------------------------------------------------------------------------

_GREP_TEXT_EXTENSIONS = {".xml", ".json", ".txt"}
_GREP_ZIP_EXTENSIONS = {".zip", ".pptx", ".docx", ".xlsx"}


def _grep_in_bytes(
    data: bytes,
    keyword: str,
    case_sensitive: bool,
    path_prefix: str,
    results: list[str],
    cap: int,
) -> None:
    """Search `data` (text file bytes) for `keyword`; append hits to `results`."""
    text = data.decode("utf-8", errors="ignore")
    needle = keyword if case_sensitive else keyword.lower()
    for line in text.splitlines():
        if len(results) >= cap:
            return
        haystack = line if case_sensitive else line.lower()
        if needle in haystack:
            trimmed = line.strip()
            if len(trimmed) > 200:
                trimmed = trimmed[:200] + "..."
            results.append(f"[{path_prefix}] {trimmed}")


def _grep_zip(
    zf: zipfile.ZipFile,
    keyword: str,
    case_sensitive: bool,
    prefix: str,
    results: list[str],
    cap: int,
) -> None:
    """Recurse into `zf`, searching text entries and nested zips."""
    for info in zf.infolist():
        if len(results) >= cap:
            return
        name = info.filename
        ext = Path(name).suffix.lower()
        full_path = f"{prefix}{name}"

        if ext in _GREP_TEXT_EXTENSIONS:
            try:
                raw = _read_capped(zf, name)
                _grep_in_bytes(raw, keyword, case_sensitive, full_path, results, cap)
            except Exception:
                pass
        elif ext in _GREP_ZIP_EXTENSIONS or ext == ".bin":
            try:
                raw = _read_capped(zf, name)
                with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                    _grep_zip(inner, keyword, case_sensitive, f"{full_path}!/", results, cap)
            except Exception:
                pass


@tool
def srdp_grep(
    zip_path: str,
    keyword: str,
    case_sensitive: bool = False,
    config: RunnableConfig = None,
) -> str:
    """Search all text entries in a SRDP zip for a keyword.

    Returns matching entry names and the line containing the match (up to 200
    chars of context per line).  Handles nested zips transparently (.bin,
    .pptx, .docx, .xlsx entries are opened and searched recursively).
    Caps output at 50 matches.

    Args:
        zip_path:       Path to the SRDP / zip file, relative to project root.
        keyword:        The string to search for.
        case_sensitive: If True, match is case-sensitive (default: False).
    """
    root = _root(config)
    p, display = _resolve_srdp(root, zip_path)
    if not _within(root, p):
        return f"Error: path escapes project root: {zip_path}"
    if not p.is_file():
        return f"Error: file not found: {zip_path}"

    results: list[str] = []
    cap = 50

    try:
        with zipfile.ZipFile(str(p)) as zf:
            _grep_zip(zf, keyword, case_sensitive, "", results, cap)
    except zipfile.BadZipFile:
        return f"Error: {zip_path} is not a valid zip file."
    except Exception as exc:
        return f"Error: {exc}"

    if not results:
        return f"No matches found for '{keyword}' in {display}."

    header = (
        f"Found {len(results)} match(es) for '{keyword}' in {display}"
        + (" (capped at 50)" if len(results) >= cap else "")
        + ":\n"
    )
    return header + "\n".join(results)
