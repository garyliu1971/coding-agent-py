"""Tests for backup/restore and diff summary output.

Covers:
- edit_file returns line-numbered diff in its output
- write_file creates a backup before overwriting
- restore_file recovers the pre-edit content
- restore_file after edit_file recovers correctly
- second restore fails (backup consumed)
- restore without prior backup gives clear error
- backup is NOT created if edit validation fails (no changes made)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from coding_agent.tools.filesystem import _bak_path, edit_file, restore_file, write_file

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(root: Path) -> dict:
    return {
        "configurable": {
            "project_root": str(root),
            "read_only": False,
            "file_read_limit": 60_000,
            "backup_dir": str(root.parent / (root.name + "_bak")),
        }
    }


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _invoke_edit(root: Path, path: str, edits: list[dict]) -> str:
    return edit_file.invoke({"path": path, "edits": edits}, config=_cfg(root))


def _invoke_write(root: Path, path: str, content: str) -> str:
    return write_file.invoke({"path": path, "content": content}, config=_cfg(root))


def _invoke_restore(root: Path, path: str) -> str:
    return restore_file.invoke({"path": path}, config=_cfg(root))


# ---------------------------------------------------------------------------
# diff output
# ---------------------------------------------------------------------------

def test_edit_returns_diff_with_line_numbers(tmp_path):
    _write(tmp_path, "a.py", "x = 1\ny = 2\nz = 3\n")
    out = _invoke_edit(tmp_path, "a.py", [{"old_string": "y = 2", "new_string": "y = 99"}])
    assert "OK" in out
    # Diff should mention the changed line
    assert "Changes:" in out
    assert "y = 2" in out   # removed line shown
    assert "y = 99" in out  # added line shown
    # Line number should appear — with context=1 the hunk spans L1-L3
    assert "L1" in out or "L2" in out


def test_edit_diff_shows_context_line(tmp_path):
    _write(tmp_path, "b.py", "a = 1\nb = 2\nc = 3\n")
    out = _invoke_edit(tmp_path, "b.py", [{"old_string": "b = 2", "new_string": "b = 20"}])
    # With context=1, the surrounding lines a=1 and c=3 should appear
    assert "a = 1" in out or "c = 3" in out


def test_multi_edit_diff_shows_both_hunks(tmp_path):
    _write(tmp_path, "c.py", "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n")
    out = _invoke_edit(tmp_path, "c.py", [
        {"old_string": "a = 1", "new_string": "a = 10"},
        {"old_string": "e = 5", "new_string": "e = 50"},
    ])
    assert "a = 10" in out
    assert "e = 50" in out


# ---------------------------------------------------------------------------
# backup created by edit_file
# ---------------------------------------------------------------------------

def test_edit_creates_backup(tmp_path):
    _write(tmp_path, "d.py", "original\n")
    _invoke_edit(tmp_path, "d.py", [{"old_string": "original", "new_string": "changed"}])
    bak = _bak_path(tmp_path.resolve(), "d.py", _cfg(tmp_path))
    assert bak.is_file(), "Backup should be created by edit_file"
    assert bak.read_text() == "original\n"


def test_write_file_creates_backup_on_overwrite(tmp_path):
    _write(tmp_path, "e.py", "v1\n")
    _invoke_write(tmp_path, "e.py", "v2\n")
    bak = _bak_path(tmp_path.resolve(), "e.py", _cfg(tmp_path))
    assert bak.is_file()
    assert bak.read_text() == "v1\n"


def test_write_file_no_backup_on_create(tmp_path):
    """New files have nothing to back up."""
    _invoke_write(tmp_path, "new.py", "hello\n")
    bak = _bak_path(tmp_path.resolve(), "new.py", _cfg(tmp_path))
    assert not bak.exists(), "No backup expected for brand-new file"


# ---------------------------------------------------------------------------
# restore_file
# ---------------------------------------------------------------------------

def test_restore_after_edit(tmp_path):
    _write(tmp_path, "f.py", "original content\n")
    _invoke_edit(tmp_path, "f.py", [{"old_string": "original content", "new_string": "wrong content"}])
    assert (tmp_path / "f.py").read_text() == "wrong content\n"

    out = _invoke_restore(tmp_path, "f.py")
    assert "OK" in out
    assert (tmp_path / "f.py").read_text() == "original content\n"


def test_restore_after_write(tmp_path):
    _write(tmp_path, "g.py", "v1\n")
    _invoke_write(tmp_path, "g.py", "v2\n")
    _invoke_restore(tmp_path, "g.py")
    assert (tmp_path / "g.py").read_text() == "v1\n"


def test_restore_consumes_backup(tmp_path):
    """After restore, a second restore should fail (backup consumed)."""
    _write(tmp_path, "h.py", "v1\n")
    _invoke_edit(tmp_path, "h.py", [{"old_string": "v1", "new_string": "v2"}])
    _invoke_restore(tmp_path, "h.py")  # first restore: OK

    out2 = _invoke_restore(tmp_path, "h.py")  # second restore: no backup
    assert "Error" in out2
    assert "no backup" in out2.lower()


def test_restore_without_prior_edit_gives_error(tmp_path):
    _write(tmp_path, "i.py", "content\n")
    out = _invoke_restore(tmp_path, "i.py")
    assert "Error" in out
    assert "no backup" in out.lower()


def test_restore_nonexistent_file_gives_error(tmp_path):
    out = _invoke_restore(tmp_path, "does_not_exist.py")
    assert "Error" in out


# ---------------------------------------------------------------------------
# backup NOT created when validation fails
# ---------------------------------------------------------------------------

def test_no_backup_when_edit_fails_validation(tmp_path):
    """If validation fails nothing is written and NO backup is taken."""
    _write(tmp_path, "j.py", "x = 1\n")
    out = _invoke_edit(tmp_path, "j.py", [{"old_string": "NOTEXIST", "new_string": "y = 2"}])
    assert "Error" in out
    # File must be unchanged
    assert (tmp_path / "j.py").read_text() == "x = 1\n"
    assert not _bak_path(tmp_path.resolve(), "j.py", _cfg(tmp_path)).exists()
