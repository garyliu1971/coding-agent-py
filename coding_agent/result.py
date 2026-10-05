"""Pure helpers behind the machine-readable run result (no model, no network).

Used by cli.py for ``run --json-result`` / ``--propose`` / ``--task-file``.

Exit codes: 0 finished | 2 usage/config error | 3 incomplete | 4 failure |
130 interrupted.
"""
from __future__ import annotations

import difflib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INCOMPLETE = 3
EXIT_FAILED = 4

# Directories never copied for --propose and never diffed.
PROPOSE_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
# Credential files: not copied into the propose temp tree, not diffed.
def _is_secret_name(fn: str) -> bool:
    n = fn.lower()
    return n == ".env" or (n.startswith(".env.") and not n.endswith((".example", ".sample", ".template", ".dist")))


def _is_link(p: Path) -> bool:
    """Symlink or Windows junction (Path.is_symlink() is False for junctions)."""
    return p.is_symlink() or getattr(os.path, "isjunction", lambda _p: False)(p)


PROPOSE_MAX_BYTES = 50 * 1024 * 1024
PROPOSE_MAX_FILES = 5000
_DIFF_MAX_FILE_BYTES = 2 * 1024 * 1024

ALLOW_WRITE_BLOCKED = "path not allowed by --allow-write"


class ProposeTooLarge(Exception):
    """The project is too big to copy for --propose."""


# ------------------------------------------------------------ status / codes --
def status_for(stop_reason: str | None, failed: bool = False) -> str:
    if failed:
        return "failed"
    return "ok" if stop_reason in ("finished", "soft_finished") else "incomplete"


def exit_code_for(status: str) -> int:
    return {"ok": EXIT_OK, "incomplete": EXIT_INCOMPLETE}.get(status, EXIT_FAILED)


def normalize_usage(usage: dict | None, wall_seconds: float = 0.0) -> dict:
    u = usage or {}
    out = {k: int(u.get(k) or 0) for k in
           ("input_tokens", "output_tokens", "total_tokens", "llm_calls", "steps")}
    out["wall_seconds"] = round(float(wall_seconds), 1)
    return out


def status_line(result: dict) -> str:
    return "status={} stop={} tokens={} files={}".format(
        result.get("status"), result.get("stop_reason"),
        (result.get("usage") or {}).get("total_tokens", 0),
        len(result.get("files_changed") or []),
    )


# ------------------------------------------------------------ files changed --
def _norm_path(path: str, root: Path | None) -> str:
    p = str(path).strip().replace("\\", "/")
    if root is not None:
        try:
            p = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    while p.startswith("./"):
        p = p[2:]
    return p


def files_changed_from_messages(messages: Iterable, root: Path | None = None) -> list[dict]:
    """[{path, action}] from write/edit/delete/move calls that SUCCEEDED ("OK..." result).

    created+modified -> created; created+deleted -> dropped; a moved file is
    reported once at its destination (with ``from``).
    """
    calls: dict[str, tuple[str, dict]] = {}
    changes: dict[str, dict] = {}

    def apply(path: str, action: str, src: str | None = None) -> None:
        prev = changes.get(path)
        if action == "modified" and prev:
            return  # keep created / moved
        if action == "deleted" and prev and prev["action"] == "created":
            del changes[path]
            return
        entry = {"path": path, "action": action}
        if src:
            entry["from"] = src
        changes.pop(path, None)  # re-insert to keep chronological order
        changes[path] = entry

    for m in messages:
        mtype = getattr(m, "type", "")
        if mtype == "ai":
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("id"):
                    calls[tc["id"]] = (tc.get("name", ""), tc.get("args") or {})
        elif mtype == "tool":
            name, args = calls.get(getattr(m, "tool_call_id", ""), (getattr(m, "name", ""), {}))
            text = m.content if isinstance(m.content, str) else str(m.content)
            if not text.startswith("OK") or "unchanged" in text[:80]:
                continue
            if name == "write_file" and args.get("path"):
                apply(_norm_path(args["path"], root),
                      "created" if "(created" in text[:200] else "modified")
            elif name in ("edit_file", "restore_file") and args.get("path"):
                apply(_norm_path(args["path"], root), "modified")
            elif name == "delete_file" and args.get("path"):
                apply(_norm_path(args["path"], root), "deleted")
            elif name == "move_file" and args.get("src") and args.get("dst"):
                src, dst = _norm_path(args["src"], root), _norm_path(args["dst"], root)
                prev = changes.pop(src, None)
                apply(dst, "created" if prev and prev["action"] == "created" else "moved", src)
    return list(changes.values())


def count_blocked_writes(messages: Iterable) -> int:
    n = 0
    for m in messages:
        if getattr(m, "type", "") == "tool":
            c = m.content if isinstance(m.content, str) else str(m.content)
            if c.startswith("Error:") and ALLOW_WRITE_BLOCKED in c[:200]:
                n += 1
    return n


# ------------------------------------------------------------------ result ---
def build_result(
    *,
    final: dict | None,
    answer: str,
    model: str,
    root: str,
    read_only: bool,
    session_log: str | None,
    wall_seconds: float,
    warnings: list[str] | None = None,
    error: str | None = None,
    diff: str | None = None,
    files_root: Path | None = None,
) -> dict:
    """Assemble the run-result dict (see README "Run result JSON")."""
    final = final or {}
    failed = error is not None
    stop_reason = final.get("stop_reason") or ("error" if failed else "no_final_answer")
    messages = final.get("messages") or []
    warns = list(warnings or [])
    blocked = count_blocked_writes(messages)
    if blocked:
        warns.append(f"{blocked} write(s) blocked by --allow-write")
    if not read_only and any(
            tc.get("name") == "run_shell"
            for m in messages if getattr(m, "type", "") == "ai"
            for tc in (getattr(m, "tool_calls", None) or [])):
        warns.append("run_shell was used: files changed by shell commands are NOT tracked in files_changed")
    status = status_for(stop_reason, failed)
    if status == "incomplete":
        warns.append(f"run did not finish normally: {stop_reason}")
    result = {
        "status": status,
        "stop_reason": stop_reason,
        "answer": answer,
        "files_changed": files_changed_from_messages(messages, files_root),
        "usage": normalize_usage(final.get("usage"), wall_seconds),
        "model": model,
        "root": root,
        "read_only": bool(read_only),
        "session_log": session_log,
        "warnings": warns,
        "error": error,
    }
    if diff is not None:
        result["diff"] = diff
    return result


def failure_result(error: str, *, model: str = "", root: str = "", read_only: bool = False,
                   session_log: str | None = None, wall_seconds: float = 0.0) -> dict:
    return build_result(final=None, answer="", model=model, root=root, read_only=read_only,
                        session_log=session_log, wall_seconds=wall_seconds, error=error)


def render_output(result: dict) -> str:
    """Markdown for --output.  Never empty."""
    answer = (result.get("answer") or "").strip()
    if not answer:
        if result.get("error"):
            answer = f"NO RESULT: run failed ({result['error']})"
        else:
            answer = f"NO RESULT: stop_reason={result.get('stop_reason')}"
    if result.get("diff") is not None:
        diff = result["diff"] or "(no changes proposed)"
        answer += "\n\n## Proposed diff\n\n```diff\n" + diff.rstrip("\n") + "\n```\n"
    return answer


def write_text_utf8(path: str | Path, text: str) -> Path:
    """UTF-8 (no BOM), LF-preserving write; creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return p


def write_json_result(path: str | Path, result: dict) -> Path:
    return write_text_utf8(path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")


# --------------------------------------------------------------- task input --
def resolve_task(words: list[str] | None, task_file: str | None, stdin=None) -> str:
    """Task text from positional words, --task-file FILE, or '-' (stdin). Raises ValueError."""
    words = list(words or [])
    stdin = stdin if stdin is not None else sys.stdin
    if words and task_file:
        raise ValueError("give the task either as words or via --task-file, not both")
    if task_file == "-" or words == ["-"]:
        buf = getattr(stdin, "buffer", None)
        raw = buf.read() if buf is not None else stdin.read()
        text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw.lstrip("﻿")
    elif task_file:
        try:
            text = Path(task_file).read_bytes().decode("utf-8-sig")
        except OSError as e:
            raise ValueError(f"cannot read --task-file {task_file}: {e}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"--task-file {task_file} is not valid UTF-8: {e}") from e
    else:
        text = " ".join(words)
    text = text.strip()
    if not text:
        raise ValueError("no task given (pass task words, --task-file FILE, or '-' for stdin)")
    return text


# ------------------------------------------------------------------ propose --
def _walk_files(root: Path):
    """Yield (relative posix path, absolute Path) for regular non-skipped files."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in PROPOSE_SKIP_DIRS and not _is_link(Path(dirpath) / d)
        )
        for fn in sorted(filenames):
            fp = Path(dirpath) / fn
            if _is_link(fp) or fn.endswith(".pyc") or _is_secret_name(fn):
                continue
            yield fp.relative_to(root).as_posix(), fp


def copy_tree_limited(src: Path, dst: Path, max_bytes: int = PROPOSE_MAX_BYTES,
                      max_files: int = PROPOSE_MAX_FILES) -> tuple[int, int]:
    """Copy ``src`` into ``dst`` skipping PROPOSE_SKIP_DIRS.  Returns (files, bytes).

    Checks the limits BEFORE copying anything; raises ProposeTooLarge.
    """
    src, dst = Path(src), Path(dst)
    entries: list[tuple[str, Path]] = []
    total = 0
    for rel, fp in _walk_files(src):
        entries.append((rel, fp))
        try:
            total += fp.stat().st_size
        except OSError:
            continue
        if len(entries) > max_files:
            raise ProposeTooLarge(
                f"--propose refused: more than {max_files} files to copy in {src} "
                f"(narrow --root to the sub-project)")
        if total > max_bytes:
            raise ProposeTooLarge(
                f"--propose refused: more than {max_bytes // (1024 * 1024)} MB to copy in {src} "
                f"(narrow --root to the sub-project)")
    dst.mkdir(parents=True, exist_ok=True)
    for rel, fp in entries:
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, target)
    return len(entries), total


def _read_text_or_none(p: Path):
    """UTF-8 text of ``p``, or None if it is binary / not UTF-8 / too large."""
    data = p.read_bytes()
    if b"\x00" in data[:8192] or len(data) > _DIFF_MAX_FILE_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def unified_diff_trees(orig: Path, modified: Path) -> tuple[str, list[str]]:
    """Unified diff (LF separators, forward-slash a/ b/ paths) of text files.

    Returns (diff, skipped) where ``skipped`` lists binary / oversized files that
    differ but are not diffed.
    """
    orig, modified = Path(orig), Path(modified)
    a_files = dict(_walk_files(orig))
    b_files = dict(_walk_files(modified))
    out: list[str] = []
    skipped: list[str] = []
    for rel in sorted(set(a_files) | set(b_files)):
        pa, pb = a_files.get(rel), b_files.get(rel)
        try:
            if pa and pb and pa.read_bytes() == pb.read_bytes():
                continue
            ta = _read_text_or_none(pa) if pa else ""
            tb = _read_text_or_none(pb) if pb else ""
        except OSError:
            skipped.append(rel)
            continue
        if ta is None or tb is None:
            skipped.append(rel)
            continue
        la, lb = ta.splitlines(keepends=True), tb.splitlines(keepends=True)
        for lines in (la, lb):
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += "\n\\ No newline at end of file\n"
        chunk = list(difflib.unified_diff(
            la, lb,
            fromfile=f"a/{rel}" if pa else "/dev/null",
            tofile=f"b/{rel}" if pb else "/dev/null",
            lineterm="\n",
        ))
        if not chunk and bool(pa) != bool(pb):  # created/deleted EMPTY file: no hunks, still a change
            chunk = [f"--- a/{rel}\n" if pa else "--- /dev/null\n",
                     f"+++ b/{rel}\n" if pb else "+++ /dev/null\n"]
        if chunk:
            out.append("".join(chunk))
    return "".join(out), skipped


def sweep_stale_temp(max_age_sec: float = 24 * 3600) -> int:
    """Delete leftover propose copies and edit backups older than ``max_age_sec``.

    A killed run (wrapper timeout) never reaches its own cleanup; this keeps the
    secrets-bearing temp copies from piling up.  Returns the number removed.
    """
    base = Path(tempfile.gettempdir())
    victims = list(base.glob("coding-agent-propose-*"))
    bak = base / "coding-agent-bak"
    if bak.is_dir():
        victims += list(bak.iterdir())
    cutoff, removed = time.time() - max_age_sec, 0
    for v in victims:
        try:
            if v.is_dir() and v.stat().st_mtime < cutoff:
                shutil.rmtree(v, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
