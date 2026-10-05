"""Static diagnostics tool: syntax / lint / type-check + conflict-marker scan.

This is a pragmatic, dependency-light alternative to a full LSP client: it
shells out to the project's linters / type-checkers (when installed) and
reports their output, plus always scans for unresolved merge-conflict markers.
It remains available in ``--read-only`` mode, where it only runs the in-memory
syntax check and the conflict-marker scan: ruff/mypy/git execute project-controlled
config or plugins, so they are skipped there.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _cfg, _root, _within
from .shell import _child_env

# Unresolved merge-conflict markers.
_CONFLICT_MARKERS = ("<<<<<<<", ">>>>>>>")

# Directories / files we never scan (VCS, venvs, build artifacts, binaries).
_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".idea",
    ".vscode",
}
_MAX_SCAN_BYTES = 1_000_000   # skip bigger files when scanning / compiling
_SECTION_CAP = 4_000          # max chars per tool section in the report
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".zip", ".pyc",
    ".pyo", ".lock", ".ipynb", ".min.js", ".map", ".woff", ".woff2", ".ttf",
}


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _cap_section(text: str, limit: int = _SECTION_CAP) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return (
        text[:head]
        + f"\n...[truncated {len(text) - limit:,} chars]...\n"
        + text[-tail:]
    )


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    # Never let child tools write bytecode; stdin is closed so nothing can block.
    env = dict(_child_env(), PYTHONDONTWRITEBYTECODE="1")  # no secrets for ruff/mypy/git
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return -1, f"(timed out after {timeout}s)"
    except OSError as e:
        return -1, f"(could not run {cmd[0]}: {e})"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return proc.returncode, _cap_section("\n".join(x for x in (out, err) if x))


def _iter_scan_files(base: Path):
    """Walk ``base`` pruning _SKIP_DIRS; yields regular files (or ``base`` itself)."""
    if base.is_file():
        yield base
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def _syntax_check(root: Path, target: Path) -> list[str]:
    """Compile every .py file in memory (nothing is written to disk).

    Returns a list whose first item is a '(N .py files checked)' summary,
    followed by one line per error.
    """
    errors: list[str] = []
    checked = 0
    for f in _iter_scan_files(target):
        if f.suffix.lower() != ".py":
            continue
        try:
            if f.stat().st_size > _MAX_SCAN_BYTES:
                continue
            src = f.read_bytes()
        except OSError:
            continue
        checked += 1
        rel = f.relative_to(root).as_posix()
        try:
            compile(src, rel, "exec", dont_inherit=True)
        except SyntaxError as e:
            errors.append(f"{rel}:{e.lineno or '?'}: SyntaxError: {e.msg}")
        except (ValueError, RecursionError, MemoryError) as e:
            errors.append(f"{rel}: cannot compile: {e}")
    errors.insert(0, f"({checked} .py files checked)")
    return errors


def _detect_language(root: Path) -> str:
    """Best-effort language detection for the project root."""
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists() or (root / "requirements.txt").exists():
        return "python"
    if (root / "package.json").exists():
        return "javascript"
    if (root / "Cargo.toml").exists():
        return "rust"
    if (root / "go.mod").exists():
        return "go"
    return "unknown"


def _scan_conflict_markers(root: Path, target: Path) -> list[str]:
    """Return a list of 'path: conflict marker' strings under ``target``."""
    hits: list[str] = []
    for p in _iter_scan_files(target):
        if p.suffix.lower() in _SKIP_SUFFIXES:
            continue
        try:
            if p.stat().st_size > _MAX_SCAN_BYTES:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for marker in _CONFLICT_MARKERS:
            if marker in text:
                hits.append(f"{p.relative_to(root)}: conflict marker {marker!r}")
                break
    return hits


@tool
def run_diagnostics(
    path: str = ".",
    timeout: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Run static diagnostics on the project and report issues (Python syntax errors, ruff/mypy lint & type errors, unresolved merge-conflict markers, git whitespace errors). Use this to locate compile/lint errors instead of guessing. `path` is relative to the project root (default '.' = whole project). Does not modify files (in read-only runs only the syntax and conflict-marker checks run)."""
    root = _root(config).resolve()
    target = (root / path).resolve()
    if not _within(root, target):
        return f"Error: path escapes project root: {path}"
    if not target.exists():
        return f"Error: path does not exist: {path}"
    t = timeout or int(_cfg(config).get("diagnostics_timeout", 120))

    lang = _detect_language(root)
    sections: list[str] = [f"Language detected: {lang}"]
    # ruff/mypy/git run project-controlled config/plugins (mypy plugins, core.fsmonitor):
    # never do that in a read-only run.
    external = not _cfg(config).get("read_only")

    if lang == "python":
        # 1) syntax check, in memory (no __pycache__ is written)
        errs = _syntax_check(root, target)
        if len(errs) == 1:
            sections.append("## python syntax (compile, in-memory)\n(clean) " + errs[0])
        else:
            sections.append(
                "## python syntax (compile, in-memory) [errors]\n"
                + _cap_section("\n".join(errs[1:]) + "\n" + errs[0])
            )

        # 2) ruff — fast linter (if installed)
        if not external:
            sections.append("## ruff / mypy\n(skipped in read-only mode)")
        elif _which("ruff"):
            rc, out = _run(["ruff", "check", "--no-cache", str(target)], root, t)
            sections.append(f"## ruff check [exit {rc}]\n{out or '(clean)'}")
        else:
            sections.append("## ruff check\n(not installed — skipped)")

        # 3) mypy — type checker (if installed)
        if not external:
            pass
        elif _which("mypy"):
            rc, out = _run(["mypy", "--cache-dir", os.devnull, str(target)], root, t)
            sections.append(f"## mypy [exit {rc}]\n{out or '(clean)'}")
        else:
            sections.append("## mypy\n(not installed — skipped)")
    else:
        sections.append(
            "## language-specific linters\n"
            f"(no linter configured for '{lang}'; install ruff/mypy for Python, "
            "or run your project's build command via run_shell)"
        )

    # 4) merge-conflict markers (language-agnostic)
    hits = _scan_conflict_markers(root, target)
    sections.append("## merge-conflict scan\n" + (_cap_section("\n".join(hits)) if hits else "(no conflict markers)"))

    # 5) git whitespace / conflict errors (language-agnostic)
    if external and (root / ".git").exists():
        rc, out = _run(["git", "--no-optional-locks", "diff", "--check"], root, t)
        sections.append(f"## git diff --check [exit {rc}]\n{out or '(clean)'}")

    return "\n\n".join(sections)
