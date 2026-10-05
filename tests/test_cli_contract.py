"""Offline tests for the sub-agent CLI contract: result JSON, exit codes, task input,
--propose copy/diff, --allow-write scoping, --no-shell, --quiet.  No model, no network."""
from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from coding_agent import cli
from coding_agent import result as R
from coding_agent.config import Config
from coding_agent.tools.filesystem import (
    delete_file, edit_file, move_file, write_file,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _call(i, name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def _res(i, name, text):
    return ToolMessage(content=text, tool_call_id=f"c{i}", name=name)


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="ca_test_"))


# ---------------------------------------------------------------- files_changed
def test_files_changed():
    print("== files_changed_from_messages ==")
    msgs = [
        HumanMessage("t"),
        _call(1, "write_file", path="a.py", content="x"),
        _res(1, "write_file", "OK: wrote a.py (created, ~1 lines added / ~0 removed)."),
        _call(2, "edit_file", path="a.py", edits=[]),
        _res(2, "edit_file", "OK: edited a.py (1 replacement applied)."),
        _call(3, "edit_file", path="b.py", edits=[]),
        _res(3, "edit_file", "OK: edited b.py (1 replacement applied)."),
        _call(4, "edit_file", path="c.py", edits=[]),
        _res(4, "edit_file", "Error: old_string not found"),
        _call(5, "write_file", path="d.py", content="x"),
        _res(5, "write_file", "OK: d.py unchanged (content identical)."),
        _call(6, "delete_file", path="e.py"),
        _res(6, "delete_file", "OK: deleted e.py"),
        _call(7, "move_file", src="f.py", dst="g.py"),
        _res(7, "move_file", "OK: moved f.py → g.py"),
        _call(8, "write_file", path="tmp.txt", content="x"),
        _res(8, "write_file", "OK: wrote tmp.txt (created, ~1 lines added / ~0 removed)."),
        _call(9, "delete_file", path="tmp.txt"),
        _res(9, "delete_file", "OK: deleted tmp.txt"),
        _call(10, "write_file", path=".\\sub\\h.py", content="x"),
        _res(10, "write_file", "OK: wrote .\\sub\\h.py (overwrote, ~1 lines added / ~0 removed)."),
    ]
    got = {(c["path"], c["action"]) for c in R.files_changed_from_messages(msgs)}
    check("created then edited stays created", ("a.py", "created") in got, str(got))
    check("edit -> modified", ("b.py", "modified") in got)
    check("failed edit ignored", not any(p == "c.py" for p, _ in got))
    check("unchanged write ignored", not any(p == "d.py" for p, _ in got))
    check("delete", ("e.py", "deleted") in got)
    check("move reported at destination", ("g.py", "moved") in got)
    check("created+deleted dropped", not any(p == "tmp.txt" for p, _ in got))
    check("backslash path normalised", ("sub/h.py", "modified") in got, str(got))
    mv = [c for c in R.files_changed_from_messages(msgs) if c["action"] == "moved"][0]
    check("move has 'from'", mv.get("from") == "f.py")


# ----------------------------------------------------------------- result dict
def _final(stop="finished", answer="done", msgs=None, usage=None):
    return {
        "messages": msgs or [AIMessage(content=answer)],
        "final_summary": answer, "stop_reason": stop,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                           "llm_calls": 2, "steps": 2},
    }


def test_result_dict():
    print("== build_result / exit codes ==")
    r = R.build_result(final=_final(), answer="done", model="m", root="/r", read_only=False,
                       session_log="/l.log", wall_seconds=1.234)
    check("ok status", r["status"] == "ok" and r["error"] is None)
    check("keys", {"status", "stop_reason", "answer", "files_changed", "usage", "model", "root",
                   "read_only", "session_log", "warnings", "error"} <= set(r))
    check("no diff key unless propose", "diff" not in r)
    check("usage has wall_seconds", r["usage"]["wall_seconds"] == 1.2 and r["usage"]["total_tokens"] == 15)
    check("json serialisable", json.loads(json.dumps(r, ensure_ascii=False))["status"] == "ok")
    check("exit ok", R.exit_code_for(r["status"]) == 0)
    r = R.build_result(final=_final(stop="soft_finished"), answer="x", model="m", root="/r",
                       read_only=False, session_log=None, wall_seconds=0)
    check("soft_finished -> ok/0", r["status"] == "ok" and R.exit_code_for(r["status"]) == 0)
    for stop in ("token_budget", "time_budget", "max_steps", "stalled", "no_final_answer"):
        r = R.build_result(final=_final(stop=stop), answer="", model="m", root="/r",
                           read_only=False, session_log=None, wall_seconds=0)
        check(f"{stop} -> incomplete/3", r["status"] == "incomplete" and R.exit_code_for(r["status"]) == 3)
        check(f"{stop} warning", any(stop in w for w in r["warnings"]))
    f = R.failure_result("boom", model="m", root="/r")
    check("failure -> failed/4", f["status"] == "failed" and R.exit_code_for("failed") == 4
          and f["error"] == "boom" and f["stop_reason"] == "error")
    check("empty usage zeros", f["usage"]["total_tokens"] == 0)
    blocked = _final(msgs=[_call(1, "write_file", path="x", content=""),
                           _res(1, "write_file", "Error: path not allowed by --allow-write: x")])
    r = R.build_result(final=blocked, answer="a", model="m", root="/r", read_only=False,
                       session_log=None, wall_seconds=0)
    check("blocked writes warning", any("blocked by --allow-write" in w for w in r["warnings"]))
    check("status_line", R.status_line({"status": "ok", "stop_reason": "finished",
                                         "usage": {"total_tokens": 7}, "files_changed": [1, 2]})
          == "status=ok stop=finished tokens=7 files=2")


def test_render_output():
    print("== render_output never empty ==")
    out = R.render_output({"answer": "", "stop_reason": "time_budget", "error": None})
    check("no answer -> explanation with stop_reason", "time_budget" in out and out.strip())
    out = R.render_output({"answer": "", "stop_reason": "error", "error": "X: y"})
    check("error explained", "X: y" in out)
    out = R.render_output({"answer": "hi", "diff": "--- a/x\n+++ b/x\n", "stop_reason": "finished"})
    check("diff appended under heading", out.startswith("hi") and "## Proposed diff" in out
          and "```diff" in out)
    out = R.render_output({"answer": "hi", "diff": "", "stop_reason": "finished"})
    check("empty diff noted", "no changes proposed" in out)


# ------------------------------------------------------------------ task input
class _FakeStdin:
    def __init__(self, data: bytes):
        self.buffer = io.BytesIO(data)


def test_resolve_task():
    print("== resolve_task ==")
    check("words joined", R.resolve_task(["do", "it"], None) == "do it")
    d = _tmp()
    f = d / "t.md"
    f.write_bytes("﻿line1\n中文 \"q\"\nline3\n".encode("utf-8"))
    t = R.resolve_task([], str(f))
    check("file utf-8 + BOM stripped, multi-line kept", t == "line1\n中文 \"q\"\nline3", repr(t))
    check("stdin via --task-file -", R.resolve_task([], "-", _FakeStdin("héllo".encode())) == "héllo")
    check("stdin via '-' word", R.resolve_task(["-"], None, _FakeStdin(b"abc\n")) == "abc")
    for name, args in (("both", (["x"], str(f))), ("none", ([], None)),
                       ("missing file", ([], str(d / "nope"))), ("empty", ([], "-"))):
        try:
            R.resolve_task(*args, stdin=_FakeStdin(b"  \n")) if name == "empty" else R.resolve_task(*args)
            check(f"{name} -> ValueError", False)
        except ValueError:
            check(f"{name} -> ValueError", True)
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------- propose
def _mk_project(d: Path):
    (d / "src").mkdir()
    (d / "src" / "a.py").write_text("one\ntwo\n", encoding="utf-8", newline="")
    (d / "src" / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    (d / "bin.dat").write_bytes(b"\x00\x01\x02")
    for skip in (".git", "node_modules", "__pycache__", ".venv"):
        (d / skip).mkdir()
        (d / skip / "junk.txt").write_text("junk")


def test_copy_and_diff():
    print("== copy_tree_limited / unified_diff_trees ==")
    src, dst = _tmp(), _tmp() / "copy"
    _mk_project(src)
    n, _ = R.copy_tree_limited(src, dst)
    check("copies 3 files, skips junk dirs", n == 3 and not (dst / ".git").exists()
          and not (dst / "node_modules").exists() and (dst / "src" / "a.py").is_file())
    diff, skipped = R.unified_diff_trees(src, dst)
    check("identical trees -> empty diff", diff == "" and skipped == [])
    (dst / "src" / "a.py").write_text("one\nTWO\nthree\n", encoding="utf-8", newline="")
    (dst / "new.py").write_text("print(1)", encoding="utf-8")
    (src / "gone.py").write_text("x\n", encoding="utf-8")
    (dst / "bin.dat").write_bytes(b"\x00\x09")
    (dst / "src" / "crlf.txt").write_bytes(b"a\r\nB\r\n")
    diff, skipped = R.unified_diff_trees(src, dst)
    check("modified file diff", "--- a/src/a.py\n+++ b/src/a.py\n" in diff and "-two\n+TWO\n+three\n" in diff, diff)
    check("new file from /dev/null", "--- /dev/null\n+++ b/new.py\n" in diff and "\\ No newline at end of file" in diff)
    check("deleted file to /dev/null", "--- a/gone.py\n+++ /dev/null\n" in diff)
    check("binary skipped + reported", "bin.dat" not in diff and skipped == ["bin.dat"])
    check("forward slashes only in headers", "\\src" not in diff and "a/src/crlf.txt" in diff)
    check("CRLF-only-line change visible", "-b\r\n+B\r\n" in diff, repr(diff[-200:]))
    # limits
    try:
        R.copy_tree_limited(src, _tmp() / "c2", max_files=2)
        check("file-count limit", False)
    except R.ProposeTooLarge as e:
        check("file-count limit", "files" in str(e))
    big = _tmp()
    (big / "x.bin").write_bytes(b"0" * 2048)
    try:
        R.copy_tree_limited(big, _tmp() / "c3", max_bytes=1024)
        check("size limit", False)
    except R.ProposeTooLarge as e:
        check("size limit", "MB" in str(e))


# ------------------------------------------------------------- execute_run etc
def _cfg(root: Path) -> Config:
    cfg = Config(project_root=root, model="fake-model", api_key="k")
    return cfg


def _fake_runner(edit_rel="src/a.py"):
    def runner(cfg, task, mode, **kw):
        p = cfg.project_root / edit_rel
        p.write_text("one\nCHANGED\n", encoding="utf-8", newline="")
        msgs = [HumanMessage(task),
                _call(1, "edit_file", path=edit_rel, edits=[]),
                _res(1, "edit_file", f"OK: edited {edit_rel} (1 replacement applied)."),
                AIMessage(content="all done")]
        return _final(msgs=msgs, answer="all done")
    return runner


def test_execute_run():
    print("== execute_run / cmd_run (fake agent) ==")
    root = _tmp()
    _mk_project(root)
    before = (root / "src" / "a.py").read_bytes()

    # normal run
    res = cli.execute_run(_cfg(root), "task", runner=_fake_runner())
    check("normal: ok + file reported", res["status"] == "ok"
          and res["files_changed"] == [{"path": "src/a.py", "action": "modified"}], str(res))
    check("normal: answer from run", res["answer"] == "all done")
    (root / "src" / "a.py").write_bytes(before)

    # propose
    res = cli.execute_run(_cfg(root), "task", runner=_fake_runner(), propose=True)
    check("propose: original untouched", (root / "src" / "a.py").read_bytes() == before)
    check("propose: diff in result", "-two\n+CHANGED\n" in res["diff"] and "a/src/a.py" in res["diff"])
    check("propose: files_changed relative", res["files_changed"][0]["path"] == "src/a.py")
    check("propose: root is the original", res["root"] == str(root.resolve()) or res["root"] == str(root))
    leftovers = [p for p in Path(tempfile.gettempdir()).glob("coding-agent-propose-*")]
    check("propose: temp copy removed", not any((p / "src" / "a.py").exists() for p in leftovers))

    # runner failures
    def boom(cfg, task, mode, **kw):
        raise RuntimeError("api down")
    res = cli.execute_run(_cfg(root), "task", runner=boom)
    check("exception -> failed", res["status"] == "failed" and "api down" in res["error"])

    def boom2(cfg, task, mode, **kw):
        raise cli.RunFailed(ValueError("bad"), _final(stop="x", usage={"total_tokens": 99}))
    res = cli.execute_run(_cfg(root), "task", runner=boom2)
    check("RunFailed keeps usage from last snapshot", res["status"] == "failed"
          and res["usage"]["total_tokens"] == 99)

    # cmd_run: outputs, exit codes, quiet
    outd = _tmp()
    out_f, js_f = outd / "o" / "out.md", outd / "r.json"
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        code = cli.cmd_run(_cfg(root), "task", output=str(out_f), json_result=str(js_f),
                           quiet=True, runner=_fake_runner())
    (root / "src" / "a.py").write_bytes(before)
    data = json.loads(js_f.read_bytes().decode("utf-8"))
    check("cmd_run ok -> 0", code == 0 and data["status"] == "ok")
    check("json no BOM", not js_f.read_bytes().startswith(b"\xef\xbb\xbf"))
    check("--output written", out_f.read_text(encoding="utf-8") == "all done")
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    check("quiet prints start + status line only", len(lines) == 2 and lines[0].startswith("coding-agent: start")
          and lines[1] == "status=ok stop=finished tokens=15 files=1", str(lines))

    def incomplete(cfg, task, mode, **kw):
        return _final(stop="token_budget", answer="")
    with contextlib.redirect_stdout(io.StringIO()):
        code = cli.cmd_run(_cfg(root), "t", output=str(out_f), json_result=str(js_f),
                           quiet=True, runner=incomplete)
    data = json.loads(js_f.read_text(encoding="utf-8"))
    check("incomplete -> 3", code == 3 and data["status"] == "incomplete"
          and data["stop_reason"] == "token_budget")
    check("--output non-empty when no answer", "token_budget" in out_f.read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        code = cli.cmd_run(_cfg(root), "t", output=str(out_f), json_result=str(js_f),
                           quiet=True, runner=boom)
    check("failure -> 4", code == 4 and json.loads(js_f.read_text(encoding="utf-8"))["status"] == "failed")

    # propose too large -> 2
    import coding_agent.result as RR
    orig_copy = RR.copy_tree_limited
    RR.copy_tree_limited = lambda s, d: orig_copy(s, d, max_files=1)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli.cmd_run(_cfg(root), "t", json_result=str(js_f), quiet=True,
                               propose=True, runner=_fake_runner())
    finally:
        RR.copy_tree_limited = orig_copy
    check("propose too large -> 2, agent not run", code == 2
          and (root / "src" / "a.py").read_bytes() == before
          and json.loads(js_f.read_text(encoding="utf-8"))["status"] == "failed")


def test_main_errors_and_help():
    print("== main(): usage errors / --help ==")
    import contextlib
    d = _tmp()
    js = d / "e.json"
    with contextlib.redirect_stdout(io.StringIO()):
        code = cli.main(["--root", str(d), "run", "--json-result", str(js)])
    check("no task -> exit 2", code == 2)
    check("usage error still writes JSON", json.loads(js.read_text(encoding="utf-8"))["status"] == "failed")
    tf = d / "t.txt"
    tf.write_text("x", encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        code = cli.main(["--root", str(d), "run", "words", "--task-file", str(tf)])
    check("words + --task-file -> exit 2", code == 2)
    for argv in (["run", "--help"], ["--root", ".", "run", "--help"]):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                cli.main(argv)
            ok = False
        except SystemExit as e:
            ok = e.code == 0
        h = buf.getvalue()
        check(f"{' '.join(argv)} works", ok and all(
            f in h for f in ("--json-result", "--task-file", "--propose", "--allow-write",
                             "--no-shell", "--max-tokens", "--timeout", "--tools", "Exit codes")))
    p = cli.build_parser().parse_args(["run", "--max-tokens", "5000", "--timeout", "30",
                                        "--allow-write", "a/**", "--allow-write", "b.py", "x"])
    check("parser flags", p.max_tokens == 5000 and p.timeout == 30 and p.allow_write == ["a/**", "b.py"])


def test_tools_mode_and_no_shell():
    print("== --tools / --no-shell ==")
    cfg = _cfg(_tmp())
    cli._apply_tools_mode(cfg, "auto", "summarize this SRDP zip")
    check("auto detects srdp", cfg.enable_srdp and not cfg.enable_vision)
    cli._apply_tools_mode(cfg, "auto", "fix typo in utils.py")
    check("auto plain task -> core", not cfg.enable_srdp and not cfg.enable_vision)
    cli._apply_tools_mode(cfg, "all", "x")
    check("all", cfg.enable_srdp and cfg.enable_vision)
    cli._apply_tools_mode(cfg, "core", "srdp image")
    check("core", not cfg.enable_srdp and not cfg.enable_vision)

    from coding_agent import graph as G
    check("shell on by default", "run_shell" in [t.name for t in G.build_tools(cfg)])
    cfg.allow_shell = False
    names = [t.name for t in G.build_tools(cfg)]
    check("no-shell drops run_shell, keeps edit_file", "run_shell" not in names and "edit_file" in names)
    from coding_agent.prompts import build_system_prompt
    check("prompt says there is no shell", "NO shell tool" in build_system_prompt(cfg, "run"))


# ------------------------------------------------------------- allow-write scope
def _wcfg(root: Path, globs):
    return {"configurable": {"project_root": str(root), "read_only": False,
                             "allow_write": globs, "backup_dir": str(root.parent / (root.name + "_bak"))}}


def test_allow_write():
    print("== --allow-write scoping (filesystem tools) ==")
    root = _tmp()
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "m.py").write_text("a = 1\n", encoding="utf-8")
    (root / "README.md").write_text("hi\n", encoding="utf-8")
    cfg = _wcfg(root, ["src/**/*.py"])
    r = write_file.invoke({"path": "src/pkg/new.py", "content": "x = 1\n"}, config=cfg)
    check("write inside glob ok", r.startswith("OK"), r)
    r = write_file.invoke({"path": "README.md", "content": "no"}, config=cfg)
    check("write outside glob blocked", r.startswith("Error") and "not allowed by --allow-write" in r, r)
    check("blocked file untouched", (root / "README.md").read_text(encoding="utf-8") == "hi\n")
    r = edit_file.invoke({"path": "README.md", "edits": [{"old_string": "hi", "new_string": "yo"}]}, config=cfg)
    check("edit outside blocked", "not allowed by --allow-write" in r, r)
    r = edit_file.invoke({"path": "src/pkg/m.py", "edits": [{"old_string": "1", "new_string": "2"}]}, config=cfg)
    check("edit inside ok", r.startswith("OK"), r)
    r = delete_file.invoke({"path": "README.md"}, config=cfg)
    check("delete outside blocked", "not allowed by --allow-write" in r and (root / "README.md").exists(), r)
    r = move_file.invoke({"src": "src/pkg/m.py", "dst": "elsewhere.py"}, config=cfg)
    check("move to disallowed dst blocked", "not allowed by --allow-write" in r
          and (root / "src" / "pkg" / "m.py").exists(), r)
    r = move_file.invoke({"src": "README.md", "dst": "src/pkg/r.py"}, config=cfg)
    check("move from disallowed src blocked", "not allowed by --allow-write" in r
          and (root / "README.md").exists(), r)
    r = move_file.invoke({"src": "src/pkg/m.py", "dst": "src/pkg/m2.py"}, config=cfg)
    check("move inside ok", r.startswith("OK"), r)
    cfg2 = _wcfg(root, ["./README.md", "docs\\*.md"])
    r = write_file.invoke({"path": "README.md", "content": "ok\n"}, config=cfg2)
    check("exact file glob with ./ prefix", r.startswith("OK"), r)
    r = write_file.invoke({"path": "../escape.txt", "content": "x"}, config=cfg2)
    check("root escape still blocked", r.startswith("Error"), r)
    r = write_file.invoke({"path": "free.txt", "content": "x"}, config=_wcfg(root, []))
    check("no globs -> unrestricted", r.startswith("OK"), r)
    r = write_file.invoke({"path": "semi.txt", "content": "x"}, config=_wcfg(root, "a/*;semi.txt"))
    check("';'-separated string accepted", r.startswith("OK"), r)


def main() -> int:
    for t in (test_files_changed, test_result_dict, test_render_output, test_resolve_task,
              test_copy_and_diff, test_execute_run, test_main_errors_and_help,
              test_tools_mode_and_no_shell, test_allow_write):
        t()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
