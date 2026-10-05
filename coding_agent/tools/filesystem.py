"""Filesystem / search tools exposed to the agent.

Tools read their runtime config (project root, limits, read-only flag) from the
injected ``RunnableConfig`` (the ``configurable`` section passed at invoke time).

edit_file design (mirrors pi / Claude approach)
------------------------------------------------
Key improvements over a naive str.replace():

1. **Multi-edit in one call** — `edits` is a list of {old_string, new_string}
   pairs applied atomically.  All matches are found against the *original* file;
   edits are then applied in reverse position order so earlier offsets stay valid.

2. **Fuzzy matching fallback** — when exact match fails we try again after:
   - Stripping trailing whitespace per line
   - Normalising smart quotes / em-dashes / non-breaking spaces to ASCII
   This catches the most common LLM failure mode: generating old_string with
   subtly wrong Unicode punctuation copied from rendered Markdown.

3. **CRLF preservation** — line endings are detected, normalised to LF for
   matching/replacement, then restored.  Windows files stay Windows files.

4. **Overlap detection** — edits that target the same region are rejected
   with a clear error rather than silently producing garbage.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel

from ..config import IGNORED_DIRS


# ---------------------------------------------------------------- helpers ---
def _cfg(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable", None) or {}


def _root(config: RunnableConfig) -> Path:
    # Resolve so _within() comparisons against resolved targets are reliable
    # (relative roots, symlinks, 8.3 short names).
    return Path(_cfg(config).get("project_root", ".")).expanduser().resolve()


# Fallbacks used when the config key is absent; keep equal to Config defaults.
_LIMIT_DEFAULTS = {"file_read_limit": 20_000, "tool_output_limit": 12_000}


def _limit(limit_key: str, config: RunnableConfig) -> int:
    return int(_cfg(config).get(limit_key, _LIMIT_DEFAULTS.get(limit_key, 20_000)))


def _cap(text: str, limit_key: str, config: RunnableConfig) -> str:
    limit = _limit(limit_key, config)
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - limit
    return (
        text[:head]
        + f"\n...[truncated {omitted:,} chars; read middle in line ranges]...\n"
        + text[-tail:]
    )


def _within(root: Path, p: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _is_link(p) -> bool:
    """Symlink or Windows junction (os.path.isjunction is 3.12+)."""
    return os.path.islink(p) or getattr(os.path, "isjunction", lambda _p: False)(p)


def _escapes_root(root: Path, p: Path) -> bool:
    """True if ``p`` is a link whose target lies outside ``root``."""
    try:
        return _is_link(p) and not _within(root, p.resolve())
    except OSError:
        return True


def _iter_files(base: Path, root: Path | None = None):
    """Walk ``base``; with ``root`` given, skip links that lead outside the root."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not (root is not None and _escapes_root(root, Path(dirpath) / d))
        ]
        for name in filenames:
            p = Path(dirpath) / name
            if root is not None and _escapes_root(root, p):
                continue
            yield p


def _is_secret_file(p: Path) -> bool:
    """.env / .env.local etc. hold credentials; templates (.env.example) are fine."""
    n = p.name.lower()
    if n == ".env":
        return True
    return n.startswith(".env.") and not n.endswith((".example", ".sample", ".template", ".dist"))


def _decode_text(data: bytes) -> str:
    """Decode bytes for display: UTF-16 by BOM, else UTF-8 (BOM dropped); lossy."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


# -------------------------------------------------------- backup / restore ---

# Backups live OUTSIDE the project root (system temp dir, one directory per
# agent process and project) so the target tree is never polluted.  Override the
# location with the ``backup_dir`` config key.
_RUN_ID = f"{os.getpid()}-{int(time.time())}"


def _bak_path(root: Path, rel_path: str, config: RunnableConfig = None) -> Path:
    base = _cfg(config).get("backup_dir")
    if base:
        base_dir = Path(base)
    else:
        tag = hashlib.sha1(str(root).encode("utf-8", "replace")).hexdigest()[:10]
        base_dir = Path(tempfile.gettempdir()) / "coding-agent-bak" / _RUN_ID / tag
    return base_dir / (rel_path + ".bak")


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def _backup(root: Path, rel_path: str, config: RunnableConfig = None) -> None:
    """Save a copy of root/rel_path into the out-of-tree backup area.

    Keeps exactly ONE backup per file: the state just before the most recent
    write.  Enough for the agent to undo one bad edit without needing git.
    ``rel_path`` must be root-relative (use ``_rel``); call only after the
    change has been validated.
    """
    src = root / rel_path
    if not src.is_file():
        return
    bak = _bak_path(root, rel_path, config)
    bak.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(bak))


# ------------------------------------------------- encoding / line endings ---

_UTF8_BOM = b"\xef\xbb\xbf"


class _NotUtf8(Exception):
    pass


def _read_text_strict(p: Path) -> tuple[str, bytes]:
    """Read ``p`` as UTF-8 WITHOUT any newline translation.

    Returns (text, bom).  ``bom`` is the 3-byte UTF-8 BOM when the file has one
    (stripped from text so it can be re-added on write), else b"".
    Raises _NotUtf8 for binary / UTF-16 / other non-UTF-8 content so callers
    refuse to write instead of silently corrupting the file.
    """
    data = p.read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        raise _NotUtf8("it looks like UTF-16 (BOM found)")
    bom = b""
    if data.startswith(_UTF8_BOM):
        bom, data = _UTF8_BOM, data[3:]
    if b"\x00" in data:
        raise _NotUtf8("it contains NUL bytes (binary or UTF-16/32)")
    try:
        return data.decode("utf-8"), bom
    except UnicodeDecodeError as e:
        raise _NotUtf8(f"it is not valid UTF-8 (byte {e.start})") from None


def _not_utf8_msg(path: str, why: str, verb: str) -> str:
    return f"Error: {path}: {why}; refusing to {verb} it (no changes made). Use run_shell for non-UTF-8 files."


def _write_text_exact(p: Path, text: str, bom: bytes = b"") -> None:
    """Write UTF-8 bytes verbatim (no newline translation)."""
    p.write_bytes(bom + text.encode("utf-8"))


# -------------------------------------------------------- edit_file helpers --

def _detect_line_ending(content: str) -> str:
    """Return '\r\n' if CRLF dominates the RAW text, else '\n'."""
    crlf = content.count("\r\n")
    lf = content.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _normalize_to_lf(text: str) -> str:
    """CRLF -> LF.  Lone CRs are left alone (they are real content)."""
    return text.replace("\r\n", "\n")


def _restore_line_endings(text: str, ending: str) -> str:
    """Convert LF text to ``ending``.  Text must not already contain CRLF."""
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


_FOLD = {}
for _c in "\u2018\u2019\u201a\u201b":
    _FOLD[ord(_c)] = "'"
for _c in "\u201c\u201d\u201e\u201f":
    _FOLD[ord(_c)] = '"'
for _c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212":
    _FOLD[ord(_c)] = "-"
for _c in "\xa0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000":
    _FOLD[ord(_c)] = " "


def _fuzzy_text(text_lf: str) -> str:
    """Fuzzy form of an LF string: fold smart quotes / dashes / odd spaces to
    ASCII and strip trailing whitespace per line (1:1 char folding, no NFKC)."""
    return "\n".join(line.translate(_FOLD).rstrip() for line in text_lf.split("\n"))


class _View:
    """LF-normalised view of raw text with a map back to raw offsets.

    ``text[i]`` came from ``raw[starts[i]:ends[i]]``.  ``starts is None`` means
    the identity map (raw has no CRLF, exact view).
    """
    __slots__ = ("text", "starts", "ends")

    def __init__(self, text: str, starts, ends):
        self.text, self.starts, self.ends = text, starts, ends

    def raw_span(self, idx: int, length: int) -> tuple[int, int]:
        if self.starts is None:
            return idx, idx + length
        return self.starts[idx], self.ends[idx + length - 1]


def _exact_view(raw: str) -> _View:
    if "\r\n" not in raw:
        return _View(raw, None, None)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i, n = 0, len(raw)
    while i < n:
        if raw.startswith("\r\n", i):
            chars.append("\n"); starts.append(i); ends.append(i + 2); i += 2
        else:
            chars.append(raw[i]); starts.append(i); ends.append(i + 1); i += 1
    return _View("".join(chars), starts, ends)


def _fuzzy_view(raw: str) -> _View:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    pending: list[tuple[str, int, int]] = []  # trailing-whitespace candidates
    i, n = 0, len(raw)
    while i < n:
        if raw.startswith("\r\n", i):
            ch, w = "\n", 2
        else:
            ch, w = raw[i].translate(_FOLD), 1
        if ch == "\n":
            pending.clear()  # trailing whitespace of this line is dropped
            chars.append(ch); starts.append(i); ends.append(i + w)
        elif ch.isspace():
            pending.append((ch, i, i + w))
        else:
            for pc, ps, pe in pending:
                chars.append(pc); starts.append(ps); ends.append(pe)
            pending.clear()
            chars.append(ch); starts.append(i); ends.append(i + w)
        i += w
    return _View("".join(chars), starts, ends)


def _find_all(hay: str, needle: str) -> list[int]:
    out, i = [], hay.find(needle)
    while i != -1:
        out.append(i)
        i = hay.find(needle, i + len(needle))
    return out


def _diff_summary(old_lf: str, new_lf: str, context: int = 1) -> str:
    """Return a compact human-readable diff with 1-based line numbers.

    Format (shown to the agent in the edit_file return value)::

        Changes:
          L5:  -  x = 1
               +  x = 99
          L12-L14: - def old_name(a, b):
                   -     return a + b
                   + def new_name(a, b, c):
                   +     return a + b + c

    Keeps `context` unchanged lines before/after each hunk so the agent can
    confirm the right location was edited.
    """
    old_lines = old_lf.splitlines()
    new_lines = new_lf.splitlines()

    # Collect changed line ranges (0-based) in old content
    # Simple LCS-free approach: mark lines that differ after alignment via
    # difflib.SequenceMatcher.
    import difflib
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    hunks: list[str] = []

    for group in matcher.get_grouped_opcodes(context):
        # group is a list of (tag, i1, i2, j1, j2) opcodes
        # Determine 1-based line range in old file for the hunk header
        first_old = group[0][1] + 1
        last_old  = group[-1][2]      # exclusive → last 1-based
        if first_old == last_old:
            hunk_header = f"  L{first_old}:"
        else:
            hunk_header = f"  L{first_old}-L{last_old}:"

        hunk_lines: list[str] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for line in old_lines[i1:i2]:
                    hunk_lines.append(f"      {line}")
            elif tag in ("replace", "delete"):
                for line in old_lines[i1:i2]:
                    hunk_lines.append(f"    - {line}")
                if tag == "replace":
                    for line in new_lines[j1:j2]:
                        hunk_lines.append(f"    + {line}")
            elif tag == "insert":
                for line in new_lines[j1:j2]:
                    hunk_lines.append(f"    + {line}")

        hunks.append(hunk_header + "\n" + "\n".join(hunk_lines))

    if not hunks:
        return ""
    return "Changes:\n" + "\n".join(hunks)


# -------------------------------------------------------- edit schema --------

class _EditEntry(BaseModel):
    old_string: str
    new_string: str


# ------------------------------------------------------------------ tools ----

_LIST_ENTRY_LIMIT = 200        # list_directory / file_search entries shown
_DEFAULT_READ_LINES = 400      # read_file window when no range is given
_GREP_LINE_CHARS = 300         # per-match line cap in grep_search
_GREP_MAX_FILE_BYTES = 100_000_000   # bigger files are skipped (and reported); smaller ones are streamed
# nested quantifier such as (a+)+ or (.*)*: catastrophic backtracking, and `re` has no timeout
_REDOS_RX = re.compile(r"\([^()]*[+*][^()]*\)\s*[+*{]")


def _grep_lines(p: Path):
    """Iterator over the lines of ``p`` (split on \n only, like read_file), or None if binary.

    UTF-8 is streamed; BOM'd UTF-16 (PowerShell 5.1 redirects) is decoded whole.
    """
    fh = open(p, "rb")
    head = fh.read(8192)
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        fh.seek(0)
        with fh:
            return iter(_decode_text(fh.read()).replace("\r\n", "\n").split("\n"))
    if b"\x00" in head:
        fh.close()
        return None
    fh.seek(0)

    def gen():
        with fh:
            for n, raw in enumerate(fh):
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if n == 0:
                    line = line.lstrip("\ufeff")
                yield line[:-1] if line.endswith("\r") else line
    return gen()

@tool
def list_directory(
    path: str = ".",
    max_depth: int = 2,
    config: RunnableConfig = None,
) -> str:
    """List the files and directories under `path` (relative to project root) up to `max_depth` levels. Use this first to understand the project structure."""
    root = _root(config)
    target = (root / path).resolve()
    if not target.exists():
        return f"Error: path does not exist: {path}"
    if not _within(root, target):
        return f"Error: path escapes project root: {path}"
    if target.is_file():
        return target.relative_to(root).as_posix()

    lines: list[str] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for e in entries:
            if e.name in IGNORED_DIRS or _escapes_root(root, e):
                continue
            rel = e.relative_to(root).as_posix()
            if e.is_dir():
                lines.append(f"{'  ' * depth}[dir ] {rel}/")
                walk(e, depth + 1)
            else:
                try:
                    size = e.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{'  ' * depth}[file] {rel}  ({size:,} B)")

    walk(target, 0)
    if not lines:
        return "(empty directory)"
    if len(lines) > _LIST_ENTRY_LIMIT:
        more = len(lines) - _LIST_ENTRY_LIMIT
        lines = lines[:_LIST_ENTRY_LIMIT] + [
            f"...{more} more entries not shown (use a deeper `path` or a smaller max_depth)"
        ]
    return "\n".join(lines)


@tool
def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Read a text file. Use `start_line`/`end_line` (1-based, inclusive) to read only a range of a large file. Output is line-numbered so you can reference lines later."""
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    if not p.is_file():
        return f"Error: not a file or does not exist: {path}"
    if _is_secret_file(p):
        return f"Error: {path} looks like a credentials file; reading it is not allowed."
    try:
        raw = _decode_text(p.read_bytes()).replace("\r\n", "\n")
    except OSError as e:
        return f"Error reading {path}: {e}"
    lines = raw.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()  # trailing newline is not an extra line
    total = len(lines)
    start = start_line or 1
    end = min(end_line or total, total)
    if start < 1 or start > total:
        return f"Error: start_line {start} out of range (file has {total} lines)."
    windowed = not start_line and not end_line and total > _DEFAULT_READ_LINES
    if windowed:
        end = _DEFAULT_READ_LINES
    # Keep whole lines only, within the char budget (reserve room for header/footer).
    budget = max(_limit("file_read_limit", config) - 200, 200)
    out: list[str] = []
    used = 0
    shown_end = start - 1
    for i in range(start, end + 1):
        row = f"{i}: {lines[i - 1]}"
        if used + len(row) + 1 > budget:
            if shown_end < start:  # a single huge line: keep a prefix of it
                out.append(row[:budget] + f"...[line {i} cut at {budget} chars]")
                shown_end = i
            break
        out.append(row)
        used += len(row) + 1
        shown_end = i
    header = f"--- {path} (lines {start}-{shown_end} of {total}) ---\n"
    footer = ""
    if shown_end < total and (windowed or shown_end < end):
        footer = (
            f"\n[showing {start}-{shown_end} of {total} lines; "
            "call read_file with start_line/end_line for more]"
        )
    return header + "\n".join(out) + footer


@tool
def grep_search(
    pattern: str,
    path: str = ".",
    is_regexp: bool = True,
    max_results: int = 50,
    config: RunnableConfig = None,
) -> str:
    """Search file contents under `path` for `pattern` (regex if `is_regexp`, else plain substring). Returns 'file:line: content' matches. Use this to find where symbols are used."""
    root = _root(config)
    target = (root / path).resolve()
    if not target.exists():
        return f"Error: path does not exist: {path}"
    if not _within(root, target):
        return f"Error: path escapes project root: {path}"
    if is_regexp and _REDOS_RX.search(pattern):
        return ("Error: pattern has a nested quantifier (e.g. '(a+)+') that can hang the search; "
                "simplify it or pass is_regexp=false.")
    try:
        rx = re.compile(pattern) if is_regexp else re.compile(re.escape(pattern))
    except re.error as e:
        return f"Error: invalid regex: {e}"

    matches: list[str] = []
    total_chars = 0
    total_cap = _limit("tool_output_limit", config)
    skipped: list[str] = []  # binary / oversized files: reported, never silently dropped

    def skip_note() -> str:
        if not skipped:
            return ""
        shown = ", ".join(skipped[:5]) + (f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else "")
        return f"\n[note: {len(skipped)} binary/oversized file(s) not searched: {shown}]"

    files = [target] if target.is_file() else list(_iter_files(target, root))
    for p in files:
        rel = p.relative_to(root).as_posix()
        if _is_secret_file(p):
            skipped.append(rel)
            continue
        try:
            if p.stat().st_size > _GREP_MAX_FILE_BYTES:
                skipped.append(rel)
                continue
            line_iter = _grep_lines(p)
        except OSError:
            continue
        if line_iter is None:  # binary
            skipped.append(rel)
            continue
        for i, line in enumerate(line_iter, 1):
            if rx.search(line):
                s = line.strip()
                if len(s) > _GREP_LINE_CHARS:
                    s = s[:_GREP_LINE_CHARS] + f"...[line truncated, {len(s)} chars total; use read_file {rel} {i}-{i}]"
                row = f"{rel}:{i}: {s}"
                if matches and total_chars + len(row) + 1 > total_cap:
                    return "\n".join(matches) + f"\n...[stopped: output cap reached after {len(matches)} matches; narrow the pattern or path]" + skip_note()
                matches.append(row)
                total_chars += len(row) + 1
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\n...[stopped at {max_results} matches]" + skip_note()
    if not matches:
        return "(no matches)" + skip_note()
    return "\n".join(matches) + skip_note()


def _glob_to_regex(glob_pattern: str) -> re.Pattern:
    """Convert a glob pattern (with `**`, `*`, `?`, and `{a,b}` braces) to a regex."""
    out: list[str] = []
    i = 0
    n = len(glob_pattern)
    while i < n:
        ch = glob_pattern[i]
        if ch == "*":
            if i + 1 < n and glob_pattern[i + 1] == "*":
                out.append(".*")
                i += 2
                if i < n and glob_pattern[i] == "/":
                    i += 1
            else:
                out.append("[^/]*")
                i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "{":
            end = glob_pattern.find("}", i)
            if end != -1:
                parts = [p for p in glob_pattern[i + 1 : end].split(",") if p]
                out.append("(?:" + "|".join(re.escape(p) for p in parts) + ")")
                i = end + 1
            else:
                out.append(re.escape(ch))
                i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _write_denied(root: Path, p: Path, config: RunnableConfig, label: str = "") -> str:
    """Error string if --allow-write globs are set and ``p`` is not covered, else "".

    Globs are matched against the root-relative forward-slash path (``**`` spans
    directories).  No globs configured -> everything inside the root is allowed.
    """
    globs = _cfg(config).get("allow_write") or []
    if isinstance(globs, str):
        globs = [g for g in globs.split(";") if g.strip()]
    if not globs:
        return ""
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        return f"Error: path not allowed by --allow-write: {label or p}"
    for g in globs:
        g = g.strip().replace("\\", "/")
        if g.startswith("./"):
            g = g[2:]
        if g and _glob_to_regex(g).match(rel):
            return ""
    return f"Error: path not allowed by --allow-write: {label or rel}"


@tool
def file_search(
    glob_pattern: str,
    path: str = ".",
    config: RunnableConfig = None,
) -> str:
    """Find files by filename glob pattern. Supports `**` (any depth), `*`, `?`, and `{a,b}` alternatives — e.g. '**/*.ts', '*service*.py', 'manifest*.xml', '*.{json,yml}'. Returns matching relative paths."""
    root = _root(config)
    target = (root / path).resolve()
    if not target.exists():
        return f"Error: path does not exist: {path}"
    if not _within(root, target):
        return f"Error: path escapes project root: {path}"
    rx = _glob_to_regex(glob_pattern)
    results = []
    files = [target] if target.is_file() else list(_iter_files(target, root))
    for p in files:
        rel = p.relative_to(root).as_posix()
        if rx.match(rel) or rx.match(p.name):
            results.append(rel)
    if not results:
        return "(no matches)"
    results.sort()
    if len(results) > _LIST_ENTRY_LIMIT:
        more = len(results) - _LIST_ENTRY_LIMIT
        results = results[:_LIST_ENTRY_LIMIT] + [f"...{more} more matches not shown (narrow the pattern or path)"]
    return "\n".join(results)


@tool
def write_file(
    path: str,
    content: str,
    config: RunnableConfig = None,
) -> str:
    """Write `content` to `path` (relative to project root), creating directories as needed. This OVERWRITES the file — only use it to create or fully rewrite a file; prefer `edit_file` for targeted changes."""
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; write_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    denied = _write_denied(root, p, config, path)
    if denied:
        return denied
    if p.is_dir():
        return f"Error: {path} is a directory."
    existed = p.exists()
    old_raw, bom, ending = "", b"", "\n"
    if existed:
        try:
            old_raw, bom = _read_text_strict(p)
        except _NotUtf8 as e:
            return _not_utf8_msg(path, str(e), "overwrite")
        except OSError as e:
            return f"Error reading {path}: {e}"
        ending = _detect_line_ending(old_raw)
        # Match the existing file's line endings (never double-convert).
        body = _restore_line_endings(_normalize_to_lf(content), ending)
    else:
        body = content  # new files: written exactly as given (LF by default)
    old = _normalize_to_lf(old_raw)
    content = _normalize_to_lf(content)
    if existed and old_raw == body:
        return f"OK: {path} unchanged (content identical)."
    p.parent.mkdir(parents=True, exist_ok=True)
    if existed:
        _backup(root, _rel(root, p), config)
    _write_text_exact(p, body, bom)
    added = len(set(content.splitlines()) - set(old.splitlines()))
    removed = len(set(old.splitlines()) - set(content.splitlines()))
    return (
        f"OK: wrote {path} ({'overwrote' if existed else 'created'}, "
        f"~{added} lines added / ~{removed} removed)."
    )


@tool
def edit_file(
    path: str,
    edits: List[_EditEntry],
    config: RunnableConfig = None,
) -> str:
    """Apply one or more precise text replacements to a file in a single call.

    `edits` is a list of {old_string, new_string} pairs.

    Rules:
    - Each old_string must appear EXACTLY ONCE in the file. Include enough
      surrounding lines to make it unique.
    - All old_strings are matched against the ORIGINAL file — not against the
      result of earlier edits in the same call.
    - Edits must not overlap; merge overlapping changes into one edit.
    - Fuzzy matching is tried automatically when exact match fails: trailing
      whitespace differences and Unicode punctuation variants (smart quotes,
      em-dashes, non-breaking spaces) are tolerated.  Only the matched span is
      rewritten; the rest of the file is left byte-for-byte untouched.
    - Line endings (CRLF / LF) and a UTF-8 BOM are preserved.  Non-UTF-8 files
      are refused rather than corrupted.

    Prefer one call with multiple edits over multiple single-edit calls when
    changing several locations in the same file.
    """
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; edit_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    denied = _write_denied(root, p, config, path)
    if denied:
        return denied
    if not p.is_file():
        return f"Error: not a file: {path}"
    if not edits:
        return "Error: edits list is empty — nothing to do."

    try:
        raw, bom = _read_text_strict(p)
    except _NotUtf8 as e:
        return _not_utf8_msg(path, str(e), "edit")
    except OSError as e:
        return f"Error reading {path}: {e}"
    ending = _detect_line_ending(raw)
    exact = _exact_view(raw)
    content = exact.text  # LF view; used for matching and diff output
    fuzzy: Optional[_View] = None

    # --- Validate all edits against the ORIGINAL content before applying any ---
    errors: list[str] = []
    # (view_start, length, new_text_in_file_endings, used_fuzzy) per edit
    plan: list[Optional[tuple[int, int, str, bool]]] = []
    for i, e in enumerate(edits):
        old_lf = _normalize_to_lf(e.old_string)
        new_lf = _normalize_to_lf(e.new_string)
        if not old_lf:
            errors.append(f"edits[{i}].old_string is empty.")
            plan.append(None)
            continue
        # Exact matches win; fuzzy is only a fallback when there is no exact hit.
        hits = _find_all(content, old_lf)
        use_fuzzy = False
        if not hits:
            if fuzzy is None:
                fuzzy = _fuzzy_view(raw)
            fo = _fuzzy_text(old_lf)
            hits = _find_all(fuzzy.text, fo) if fo else []
            use_fuzzy = True
            length = len(fo)
        else:
            length = len(old_lf)
        if not hits:
            errors.append(
                f"edits[{i}].old_string not found in {path}. "
                "Use read_file to check exact content and whitespace."
            )
            plan.append(None)
        elif len(hits) > 1:
            errors.append(
                f"edits[{i}].old_string found {len(hits)} times in {path}. "
                "Add more surrounding context to make it unique."
            )
            plan.append(None)
        else:
            view = fuzzy if use_fuzzy else exact
            start, end = view.raw_span(hits[0], length)
            plan.append((start, end, _restore_line_endings(new_lf, ending), use_fuzzy))
    if errors:
        return "Error — no changes made:\n" + "\n".join(f"  • {e}" for e in errors)

    # --- Check for overlaps (raw offsets of the original file) ---
    spans = sorted(((pl[0], pl[1], i) for i, pl in enumerate(plan) if pl), key=lambda x: x[0])
    for j in range(1, len(spans)):
        if spans[j - 1][1] > spans[j][0]:
            return (
                f"Error: edits[{spans[j - 1][2]}] and edits[{spans[j][2]}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    # --- Apply right-to-left on the raw text: only matched spans change ---
    result_raw = raw
    for start, end, new_text, _fz in sorted((pl for pl in plan if pl), key=lambda x: -x[0]):
        result_raw = result_raw[:start] + new_text + result_raw[end:]

    if result_raw == raw:
        return f"Error: replacements produced identical content in {path} — no changes made."

    _backup(root, _rel(root, p), config)
    _write_text_exact(p, result_raw, bom)

    diff = _diff_summary(content, _normalize_to_lf(result_raw), context=1)
    n = len(edits)
    fuzzy_n = sum(1 for pl in plan if pl and pl[3])
    header = f"OK: edited {path} ({n} replacement{'s' if n > 1 else ''} applied"
    if fuzzy_n:
        header += f"; {fuzzy_n} via fuzzy whitespace/punctuation match"
    header += ")."
    return header + ("\n" + diff if diff else "")


@tool
def delete_file(
    path: str,
    config: RunnableConfig = None,
) -> str:
    """Delete the file at `path` (relative to project root). Use sparingly."""
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; delete_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    denied = _write_denied(root, p, config, path)
    if denied:
        return denied
    if p.is_file():
        p.unlink()
        return f"OK: deleted {path}"
    return f"Error: not a file: {path}"


@tool
def restore_file(
    path: str,
    config: RunnableConfig = None,
) -> str:
    """Restore a file to its state before the last write_file or edit_file call.

    Every write_file / edit_file automatically saves a backup (outside the
    project, in the system temp dir) before overwriting.  Call this tool to
    undo a bad edit.  Only ONE level of undo is available per file.

    Use this when:
    - An edit produced incorrect code and you want to start over.
    - write_file overwrote the wrong content.
    """
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; restore_file is disabled."
    root = _root(config)
    p = (root / path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {path}"
    denied = _write_denied(root, p, config, path)
    if denied:
        return denied
    bak = _bak_path(root, _rel(root, p), config)
    if not bak.is_file():
        return (
            f"Error: no backup found for {path}. "
            "restore_file only works after a write_file or edit_file call."
        )
    shutil.copy2(str(bak), str(p))
    bak.unlink()  # consume the backup so a second restore doesn't double-undo
    return f"OK: restored {path} from backup (backup consumed — further restore not possible)."


@tool
def move_file(
    src: str,
    dst: str,
    config: RunnableConfig = None,
) -> str:
    """Move or rename a file within the project root.

    Both `src` and `dst` are relative to the project root.
    Parent directories of `dst` are created automatically.
    Use this when refactoring requires reorganising files or renaming modules.
    Remember to update import paths in other files with edit_file after moving.
    """
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; move_file is disabled."
    root = _root(config)
    s = (root / src).resolve()
    d = (root / dst).resolve()
    if not _within(root, s):
        return f"Error: src escapes project root: {src}"
    if not _within(root, d):
        return f"Error: dst escapes project root: {dst}"
    denied = _write_denied(root, s, config, src) or _write_denied(root, d, config, dst)
    if denied:
        return denied
    if not s.exists():
        return f"Error: src does not exist: {src}"
    if d.exists():
        return f"Error: dst already exists: {dst}. Delete it first to overwrite."
    if s.is_dir():
        if _cfg(config).get("allow_write"):
            return f"Error: {src} is a directory; with --allow-write only single files can be moved."
        if _within(s, d):
            return f"Error: cannot move a directory into itself: {src} -> {dst}"
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    except (OSError, shutil.Error) as e:
        return f"Error: move failed: {e}"
    return f"OK: moved {src} → {dst}"
