"""Offline tests for the tool-hardening pass (filesystem / shell / diagnostics).

Covers: line-ending + encoding preservation, out-of-tree backups, read-only
diagnostics, span-only fuzzy edits, run_shell safety, and output caps.

Runs under pytest (uses ``tmp_path``) or as a plain script:
    $env:PYTHONPATH='.'; .\\.venv\\Scripts\\python.exe tests\\test_tool_hardening.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from coding_agent.tools import filesystem as fs
from coding_agent.tools.diagnostics import run_diagnostics
from coding_agent.tools.filesystem import (
    edit_file,
    file_search,
    grep_search,
    list_directory,
    read_file,
    restore_file,
    write_file,
)
from coding_agent.tools.shell import _cap_output, _child_env, _dangerous_reason, run_shell


# ---------------------------------------------------------------- helpers ---
def _c(root: Path, **extra) -> dict:
    cfg = {
        "project_root": str(root),
        "read_only": False,
        "backup_dir": str(root.parent / (root.name + "_bak")),
    }
    cfg.update(extra)
    return {"configurable": cfg}


def _put(root: Path, rel: str, data) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return p


def _edit(root: Path, path: str, edits: list, **extra) -> str:
    return edit_file.invoke({"path": path, "edits": edits}, config=_c(root, **extra))


def _write(root: Path, path: str, content: str, **extra) -> str:
    return write_file.invoke({"path": path, "content": content}, config=_c(root, **extra))


def _snapshot(root: Path) -> dict:
    snap = {}
    for dp, dns, fns in os.walk(root):
        for d in dns:
            snap[str((Path(dp) / d).relative_to(root))] = None
        for f in fns:
            q = Path(dp) / f
            snap[str(q.relative_to(root))] = q.read_bytes()
    return snap


# ------------------------------------------------------------ line endings ---
def test_edit_lf_stays_lf(tmp_path):
    p = _put(tmp_path, "a.txt", b"one\ntwo\nthree\n")
    out = _edit(tmp_path, "a.txt", [{"old_string": "two", "new_string": "2\nTWO"}])
    assert out.startswith("OK"), out
    assert p.read_bytes() == b"one\n2\nTWO\nthree\n"


def test_edit_crlf_stays_crlf_no_double_cr(tmp_path):
    p = _put(tmp_path, "a.txt", b"one\r\ntwo\r\nthree\r\n")
    out = _edit(tmp_path, "a.txt", [{"old_string": "two", "new_string": "2\nTWO"}])
    assert out.startswith("OK"), out
    data = p.read_bytes()
    assert data == b"one\r\n2\r\nTWO\r\nthree\r\n"
    assert b"\r\r" not in data


def test_edit_multiline_old_string_on_crlf_file(tmp_path):
    p = _put(tmp_path, "a.txt", b"a\r\nb\r\nc\r\n")
    out = _edit(tmp_path, "a.txt", [{"old_string": "a\nb", "new_string": "X"}])
    assert out.startswith("OK"), out
    assert p.read_bytes() == b"X\r\nc\r\n"


def test_edit_mixed_endings_untouched_lines_byte_identical(tmp_path):
    p = _put(tmp_path, "m.txt", b"a\r\nb\r\nc\nd\r\n")
    out = _edit(tmp_path, "m.txt", [{"old_string": "c", "new_string": "C"}])
    assert out.startswith("OK"), out
    assert p.read_bytes() == b"a\r\nb\r\nC\nd\r\n"


def test_write_new_file_is_lf(tmp_path):
    out = _write(tmp_path, "sub/new.txt", "x\ny\n")
    assert out.startswith("OK"), out
    assert (tmp_path / "sub" / "new.txt").read_bytes() == b"x\ny\n"


def test_write_overwrite_keeps_existing_style(tmp_path):
    crlf = _put(tmp_path, "c.txt", b"1\r\n2\r\n")
    lf = _put(tmp_path, "l.txt", b"1\n2\n")
    _write(tmp_path, "c.txt", "1\n2\n3\n")
    _write(tmp_path, "l.txt", "1\n2\n3\n")
    assert crlf.read_bytes() == b"1\r\n2\r\n3\r\n"
    assert lf.read_bytes() == b"1\n2\n3\n"


def test_write_content_with_crlf_into_crlf_file_not_doubled(tmp_path):
    p = _put(tmp_path, "c.txt", b"1\r\n")
    _write(tmp_path, "c.txt", "1\r\n2\r\n")
    assert p.read_bytes() == b"1\r\n2\r\n"


def test_read_edit_write_round_trip(tmp_path):
    for raw in (b"alpha\nbeta\ngamma\n", b"alpha\r\nbeta\r\ngamma\r\n"):
        p = _put(tmp_path, "r.txt", raw)
        shown = read_file.invoke({"path": "r.txt"}, config=_c(tmp_path))
        body = [ln.split(": ", 1)[1] for ln in shown.split("\n")[1:] if ": " in ln]
        assert body == ["alpha", "beta", "gamma"]
        _edit(tmp_path, "r.txt", [{"old_string": "beta", "new_string": "BETA"}])
        again = p.read_bytes()
        assert again == raw.replace(b"beta", b"BETA")
        # write the same content back through write_file: must be a no-op
        out = _write(tmp_path, "r.txt", "alpha\nBETA\ngamma\n")
        assert "unchanged" in out, out
        assert p.read_bytes() == again


# ---------------------------------------------------------------- encoding ---
def test_edit_refuses_non_utf8(tmp_path):
    raw = "caf\xe9 = 1\n".encode("latin-1")
    p = _put(tmp_path, "l1.txt", raw)
    out = _edit(tmp_path, "l1.txt", [{"old_string": "= 1", "new_string": "= 2"}])
    assert out.startswith("Error") and "UTF-8" in out, out
    assert p.read_bytes() == raw


def test_write_refuses_to_overwrite_non_utf8(tmp_path):
    raw = "caf\xe9\n".encode("latin-1")
    p = _put(tmp_path, "l1.txt", raw)
    out = _write(tmp_path, "l1.txt", "new\n")
    assert out.startswith("Error"), out
    assert p.read_bytes() == raw


def test_refuses_utf16_and_binary(tmp_path):
    u16 = "hello\n".encode("utf-16")
    p1 = _put(tmp_path, "u16.txt", u16)
    p2 = _put(tmp_path, "bin.dat", b"ab\x00cd\n")
    assert _edit(tmp_path, "u16.txt", [{"old_string": "hello", "new_string": "x"}]).startswith("Error")
    assert _edit(tmp_path, "bin.dat", [{"old_string": "ab", "new_string": "x"}]).startswith("Error")
    assert p1.read_bytes() == u16 and p2.read_bytes() == b"ab\x00cd\n"


def test_utf8_bom_preserved(tmp_path):
    bom = b"\xef\xbb\xbf"
    p = _put(tmp_path, "b.txt", bom + "h\u00e9llo\r\nworld\r\n".encode("utf-8"))
    assert _edit(tmp_path, "b.txt", [{"old_string": "world", "new_string": "there"}]).startswith("OK")
    assert p.read_bytes() == bom + "h\u00e9llo\r\nthere\r\n".encode("utf-8")
    assert _write(tmp_path, "b.txt", "new\n").startswith("OK")
    assert p.read_bytes() == bom + b"new\r\n"


# ----------------------------------------------------------------- backups ---
def test_no_backup_dir_in_project_root(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    _put(proj, "a.txt", "x\n")
    cfg = {"configurable": {"project_root": str(proj), "read_only": False}}  # default backup location
    edit_file.invoke({"path": "a.txt", "edits": [{"old_string": "x", "new_string": "y"}]}, config=cfg)
    write_file.invoke({"path": "a.txt", "content": "z\n"}, config=cfg)
    assert sorted(os.listdir(proj)) == ["a.txt"]
    out = restore_file.invoke({"path": "a.txt"}, config=cfg)
    assert out.startswith("OK"), out
    assert (proj / "a.txt").read_text() == "y\n"


def test_backup_with_absolute_path_stays_out_of_tree(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    p = _put(proj, "a.txt", "x\n")
    cfg = _c(proj)
    out = edit_file.invoke(
        {"path": str(p), "edits": [{"old_string": "x", "new_string": "y"}]}, config=cfg
    )
    assert out.startswith("OK"), out
    assert sorted(os.listdir(proj)) == ["a.txt"]  # no .bak next to the source
    assert restore_file.invoke({"path": str(p)}, config=cfg).startswith("OK")
    assert p.read_text() == "x\n"


def test_no_backup_when_validation_fails(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _put(root, "a.txt", "x\nx\n")
    bak_root = Path(_c(root)["configurable"]["backup_dir"])
    assert _edit(root, "a.txt", [{"old_string": "x", "new_string": "y"}]).startswith("Error")  # ambiguous
    assert _edit(root, "a.txt", [{"old_string": "nope", "new_string": "y"}]).startswith("Error")
    assert not bak_root.exists()


# ------------------------------------------------------------- diagnostics ---
def test_run_diagnostics_leaves_target_byte_identical(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _put(root, "pyproject.toml", "[project]\nname='x'\n")
    _put(root, "pkg/__init__.py", "")
    _put(root, "pkg/mod.py", "def f():\n    return 1\n")
    _put(root, "pkg/bad.py", "def g(:\n")
    before = _snapshot(root)
    out = run_diagnostics.invoke({"path": "."}, config=_c(root))
    assert "bad.py" in out and "SyntaxError" in out, out
    assert _snapshot(root) == before  # no __pycache__, .ruff_cache, .mypy_cache ...


def test_run_diagnostics_confined_to_root(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    out = run_diagnostics.invoke({"path": ".."}, config=_c(root))
    assert out.startswith("Error") and "escapes" in out, out


# ---------------------------------------------------------- fuzzy edit_file ---
def test_fuzzy_edit_touches_only_matched_span(tmp_path):
    src = "say \u201chi\u201d   \nkeep \u2014 this  \nname = 'a'\n"
    p = _put(tmp_path, "f.txt", src)
    out = _edit(tmp_path, "f.txt", [{"old_string": "name = 'a'", "new_string": "name = 'b'"}])
    assert out.startswith("OK"), out
    assert p.read_text(encoding="utf-8") == src.replace("'a'", "'b'")


def test_fuzzy_match_smart_quotes_replaces_only_span(tmp_path):
    src = "x = \u201chello\u201d  # note  \ny = \u2014 1  \n"
    p = _put(tmp_path, "f.txt", src)
    out = _edit(tmp_path, "f.txt", [{"old_string": 'x = "hello"', "new_string": "x = 'bye'"}])
    assert out.startswith("OK") and "fuzzy" in out, out
    # only the matched span changed; the comment, trailing spaces and the em dash line survive
    assert p.read_text(encoding="utf-8") == "x = 'bye'  # note  \ny = \u2014 1  \n"


def test_exact_unique_match_not_rejected_by_fuzzy_duplicates(tmp_path):
    p = _put(tmp_path, "f.txt", "a\u2014b\na-b\n")
    out = _edit(tmp_path, "f.txt", [{"old_string": "a-b", "new_string": "Z"}])
    assert out.startswith("OK"), out
    assert p.read_text(encoding="utf-8") == "a\u2014b\nZ\n"


def test_multi_edit_is_atomic(tmp_path):
    p = _put(tmp_path, "f.txt", "one\ntwo\n")
    out = _edit(tmp_path, "f.txt", [
        {"old_string": "one", "new_string": "1"},
        {"old_string": "missing", "new_string": "x"},
    ])
    assert out.startswith("Error"), out
    assert p.read_bytes() == b"one\ntwo\n"


def test_edits_match_original_not_previous_result(tmp_path):
    p = _put(tmp_path, "f.txt", "a  \nb\n")  # trailing spaces force the fuzzy path
    out = _edit(tmp_path, "f.txt", [
        {"old_string": "a", "new_string": "b"},
        {"old_string": "b", "new_string": "c"},
    ])
    assert out.startswith("OK"), out
    assert p.read_bytes() == b"b  \nc\n"


# ------------------------------------------------------------------- shell ---
def _py_cmd(code: str) -> str:
    exe = sys.executable
    return f'& "{exe}" -c "{code}"' if os.name == "nt" else f'"{exe}" -c "{code}"'


def test_shell_stdin_not_inherited(tmp_path):
    t0 = time.time()
    out = run_shell.invoke({"command": _py_cmd("import sys; print(len(sys.stdin.read()))"), "timeout": 30}, config=_c(tmp_path))
    assert "(exit code 0)" in out and out.rstrip().endswith("0"), out
    assert time.time() - t0 < 25


def test_shell_secrets_stripped_from_env(tmp_path):
    os.environ["AZURE_TEST_VALUE"] = "azure-secret-123"
    os.environ["MY_API_TOKEN"] = "token-secret-456"
    os.environ["HARMLESS_VAR"] = "visible-789"
    try:
        env = _child_env()
        assert "AZURE_TEST_VALUE" not in env and "MY_API_TOKEN" not in env
        assert env.get("HARMLESS_VAR") == "visible-789"
        cmd = (
            'Write-Output "[$env:AZURE_TEST_VALUE][$env:MY_API_TOKEN][$env:HARMLESS_VAR]"'
            if os.name == "nt"
            else 'echo "[$AZURE_TEST_VALUE][$MY_API_TOKEN][$HARMLESS_VAR]"'
        )
        out = run_shell.invoke({"command": cmd}, config=_c(tmp_path))
        assert "secret" not in out and "visible-789" in out, out
    finally:
        for k in ("AZURE_TEST_VALUE", "MY_API_TOKEN", "HARMLESS_VAR"):
            os.environ.pop(k, None)


def test_shell_workdir_confined(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for wd in ("..", str(tmp_path)):
        out = run_shell.invoke({"command": "echo hi", "workdir": wd}, config=_c(root))
        assert out.startswith("Error") and "escapes" in out, out
    (root / "sub").mkdir()
    assert "(exit code 0)" in run_shell.invoke({"command": "echo hi", "workdir": "sub"}, config=_c(root))


def test_shell_utf8_output(tmp_path):
    cmd = 'Write-Output "caf\u00e9 \u4e2d\u6587"' if os.name == "nt" else 'printf "caf\\xc3\\xa9 \\xe4\\xb8\\xad\\xe6\\x96\\x87"'
    out = run_shell.invoke({"command": cmd}, config=_c(tmp_path))
    assert "caf\u00e9 \u4e2d\u6587" in out, out


def test_shell_timeout_kills_tree_and_returns(tmp_path):
    inner = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);print('started',flush=True);time.sleep(120)"
    t0 = time.time()
    out = run_shell.invoke({"command": _py_cmd(inner), "timeout": 3}, config=_c(tmp_path))
    assert time.time() - t0 < 40, "timeout must not hang"
    assert "timed out" in out, out


def test_shell_timeout_is_clamped(tmp_path):
    # a huge model-supplied timeout must not exceed the hard maximum
    from coding_agent.tools import shell
    assert shell._MAX_TIMEOUT <= 600


def test_shell_guard_extra_patterns():
    assert _dangerous_reason("rm -rf /*")
    assert _dangerous_reason("Remove-Item -Recurse -Force C:\\Users")
    assert _dangerous_reason("rd /s /q C:\\Windows")
    assert _dangerous_reason("git clean -fdx")
    assert _dangerous_reason("Remove-Item -Recurse -Force C:\\Users\\bob\\proj\\build") is None
    assert _dangerous_reason("git status") is None


def test_shell_output_cap_keeps_whole_lines_and_says_so():
    out = "\n".join(f"line {i:04d}" for i in range(2000))
    capped = _cap_output(out, 2000)
    assert len(capped) < len(out)
    assert "line 0000" in capped and "line 1999" in capped
    assert "output truncated: kept first" in capped and "of 2000 lines" in capped
    for ln in capped.split("\n"):
        assert ln.startswith("line ") or ln.startswith("...[output truncated")


# ------------------------------------------------------------- output caps ---
def test_read_file_default_window_and_footer(tmp_path):
    _put(tmp_path, "big.txt", "".join(f"l{i}\n" for i in range(1, 1001)))
    out = read_file.invoke({"path": "big.txt"}, config=_c(tmp_path, file_read_limit=60_000))
    assert "400: l400" in out and "401: l401" not in out
    assert out.endswith("[showing 1-400 of 1000 lines; call read_file with start_line/end_line for more]")


def test_read_file_footer_survives_char_cap_and_cuts_whole_lines(tmp_path):
    _put(tmp_path, "w.txt", "".join(f"{'x' * 80}{i}\n" for i in range(1, 301)))
    out = read_file.invoke({"path": "w.txt"}, config=_c(tmp_path, file_read_limit=2_000))
    assert len(out) <= 2_000 + 200
    last_line = out.rsplit("\n", 1)[0].rsplit("\n", 1)[-1]
    assert last_line.endswith(str(int(last_line.split(":")[0])))  # last shown line is complete
    n = int(last_line.split(":")[0])
    assert f"[showing 1-{n} of 300 lines;" in out
    assert out.endswith("for more]")


def test_read_file_explicit_range_and_small_file_have_no_footer(tmp_path):
    _put(tmp_path, "big.txt", "".join(f"l{i}\n" for i in range(1, 1001)))
    out = read_file.invoke({"path": "big.txt", "start_line": 500, "end_line": 510}, config=_c(tmp_path))
    assert "500: l500" in out and "510: l510" in out and "[showing" not in out
    _put(tmp_path, "s.txt", "a\nb\n")
    small = read_file.invoke({"path": "s.txt"}, config=_c(tmp_path))
    assert "lines 1-2 of 2" in small and "[showing" not in small


def test_fallback_defaults_match_config_defaults():
    assert fs._limit("file_read_limit", {}) == 20_000
    assert fs._limit("tool_output_limit", {}) == 12_000
    from coding_agent.tools import srdp
    assert srdp._cap_text.__defaults__ == (20_000,)
    assert srdp._read_entry_text.__defaults__ == (20_000,)
    big = "y" * 50_000
    assert len(fs._cap(big, "file_read_limit", {})) < 21_000


def test_grep_defaults_and_caps(tmp_path):
    _put(tmp_path, "many.txt", "".join(f"hit {i}\n" for i in range(200)))
    out = grep_search.invoke({"pattern": "hit"}, config=_c(tmp_path))
    assert out.count("many.txt:") == 50 and "stopped at 50 matches" in out
    _put(tmp_path, "long.txt", "needle " + "z" * 1000 + "\n")
    out = grep_search.invoke({"pattern": "needle", "path": "long.txt"}, config=_c(tmp_path))
    assert "line truncated" in out and len(out) < 500
    _put(tmp_path, "bin.dat", b"needle\x00\x01\x02")
    out = grep_search.invoke({"pattern": "needle", "path": "."}, config=_c(tmp_path))
    assert "not searched: bin.dat" in out  # skipped files are reported


def test_grep_total_output_cap(tmp_path):
    _put(tmp_path, "t.txt", "".join(f"match {'q' * 250} {i}\n" for i in range(60)))
    out = grep_search.invoke({"pattern": "match", "max_results": 60}, config=_c(tmp_path, tool_output_limit=3_000))
    assert len(out) < 3_500 and "output cap reached" in out


def test_list_directory_and_file_search_entry_cap(tmp_path):
    for i in range(250):
        _put(tmp_path, f"d/f{i:03d}.txt", "x")
    out = list_directory.invoke({"path": "d"}, config=_c(tmp_path))
    assert len(out.split("\n")) == 201 and "...50 more entries" in out
    out = file_search.invoke({"glob_pattern": "**/*.txt"}, config=_c(tmp_path))
    assert "...50 more matches" in out


def test_root_is_resolved():
    cwd = os.getcwd()
    assert fs._root({"configurable": {"project_root": "."}}) == Path(cwd).resolve()


# ------------------------------------------------------------- script mode ---
if __name__ == "__main__":
    import inspect
    import shutil
    import tempfile
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        tmp = None
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                tmp = Path(tempfile.mkdtemp(prefix="hardening_")).resolve()
                fn(tmp)
            else:
                fn()
            passed += 1
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.rmtree(str(tmp) + "_bak", ignore_errors=True)
    print(f"\n== RESULT: {passed} passed, {failed} failed ==")
    raise SystemExit(1 if failed else 0)
