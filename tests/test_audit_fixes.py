"""Offline regression tests for the second audit round (no model, no network):
main() 'run' path, --quiet errors, shell gating, soft finish, read-only diagnostics,
grep/read hardening, propose copy hygiene, zip-bomb cap, partial usage on failure."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from langchain_core.messages import AIMessage

from coding_agent import cli
from coding_agent import result as R
from coding_agent.config import Config
from coding_agent.tools import diagnostics as D
from coding_agent.tools import shell as SH
from coding_agent.tools import srdp as S
from coding_agent.tools.filesystem import grep_search, list_directory, move_file, read_file

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="audit-fix-"))


def _c(root: Path, **kw) -> dict:
    return {"configurable": {"project_root": str(root), **kw}}


def _final(stop="finished", answer="done"):
    return {"messages": [AIMessage(content=answer)], "final_summary": answer, "stop_reason": stop,
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "llm_calls": 1, "steps": 1}}


# ------------------------------------------------------------------ cli main
def test_main_run_paths():
    print("== main(run): no NameError, shell gating, quiet errors ==")
    root = _tmp()
    seen = {}

    def fake(cfg, task, mode, **kw):
        seen["shell"], seen["root"] = cfg.allow_shell, cfg.project_root
        return _final()

    old_run, old_env = cli.run_agent, {k: os.environ.get(k) for k in ("DEEPSEEK_API_KEY", "CODING_AGENT_ALLOW_WRITE")}
    os.environ["DEEPSEEK_API_KEY"] = "x"
    os.environ.pop("CODING_AGENT_ALLOW_WRITE", None)
    cli.run_agent = fake
    try:
        def run(*extra):
            seen.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                return cli.main(["--root", str(root), "run", "--quiet", *extra, "hi"])
        code = run()
        check("run without --allow-write works (os imported)", code == 0, str(code))
        check("shell on by default", seen.get("shell") is True)
        run("--no-shell")
        check("--no-shell", seen.get("shell") is False)
        run("--allow-write", "a.txt")
        check("--allow-write disables shell", seen.get("shell") is False)
        run("--propose")
        check("--propose disables shell", seen.get("shell") is False)
        os.environ["CODING_AGENT_ALLOW_WRITE"] = "x/**"
        run()
        check("env allow-write disables shell", seen.get("shell") is False)
        os.environ.pop("CODING_AGENT_ALLOW_WRITE")

        # --quiet must not swallow early errors
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["--root", str(root / "missing"), "run", "--quiet", "hi"])
        check("quiet + bad root -> exit 2 with a stderr message", code == 2 and "not a directory" in err.getvalue(),
              err.getvalue())
    finally:
        cli.run_agent = old_run
        cli.console.quiet = False
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(root, ignore_errors=True)


def test_execute_run_propose_disables_shell():
    print("== execute_run(propose) ==")
    root = _tmp()
    (root / "a.txt").write_text("x\n", encoding="utf-8")
    seen = {}

    def fake(cfg, task, mode, **kw):
        seen["shell"], seen["root"] = cfg.allow_shell, cfg.project_root
        (cfg.project_root / "empty.txt").write_text("", encoding="utf-8")
        (cfg.project_root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        return _final()

    cfg = Config(project_root=root)
    res = cli.execute_run(cfg, "t", runner=fake, propose=True)
    check("propose run has no shell", seen["shell"] is False)
    check("caller's cfg untouched", cfg.allow_shell is True)
    check("new EMPTY file appears in the diff", "+++ b/empty.txt" in res["diff"], res["diff"])
    check("secret files stay out of the diff", ".env" not in res["diff"])
    shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------- result / status
def test_soft_finish_and_shell_warning():
    print("== soft finish / run_shell warning ==")
    r = R.build_result(final=_final(stop="soft_finished", answer="STATUS: done"), answer="STATUS: done",
                       model="m", root="/r", read_only=False, session_log=None, wall_seconds=0)
    check("soft_finished -> ok, no incomplete warning", r["status"] == "ok"
          and not any("did not finish" in w for w in r["warnings"]))
    msgs = [AIMessage(content="", tool_calls=[{"name": "run_shell", "args": {"command": "ls"}, "id": "1"}])]
    f = {**_final(), "messages": msgs}
    r = R.build_result(final=f, answer="a", model="m", root="/r", read_only=False, session_log=None, wall_seconds=0)
    check("shell use is flagged", any("run_shell was used" in w for w in r["warnings"]))
    r = R.build_result(final=f, answer="a", model="m", root="/r", read_only=True, session_log=None, wall_seconds=0)
    check("no flag in read-only", not any("run_shell" in w for w in r["warnings"]))


def test_sweep_and_copy_hygiene():
    print("== sweep_stale_temp / copy hygiene ==")
    old = Path(tempfile.mkdtemp(prefix="coding-agent-propose-"))
    fresh = Path(tempfile.mkdtemp(prefix="coding-agent-propose-"))
    past = time.time() - 3 * 24 * 3600
    os.utime(old, (past, past))
    R.sweep_stale_temp(24 * 3600)
    check("stale propose dir removed, fresh kept", not old.exists() and fresh.exists())
    shutil.rmtree(fresh, ignore_errors=True)

    src, dst = _tmp(), _tmp()
    (src / ".env").write_text("K=1", encoding="utf-8")
    (src / ".env.example").write_text("K=", encoding="utf-8")
    (src / "a.py").write_text("x", encoding="utf-8")
    R.copy_tree_limited(src, dst)
    check(".env not copied, .env.example and code are", not (dst / ".env").exists()
          and (dst / ".env.example").exists() and (dst / "a.py").exists())
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(dst, ignore_errors=True)


# ------------------------------------------------------------------ tools
def test_diagnostics_read_only_and_env():
    print("== diagnostics read-only / child env ==")
    root = _tmp()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "m.py").write_text("x = 1\n", encoding="utf-8")
    calls = []
    orig = D._run
    D._run = lambda cmd, cwd, t: (calls.append(cmd[0]) or (0, ""))
    try:
        out = D.run_diagnostics.invoke({}, config=_c(root, read_only=True))
        check("read-only: no ruff/mypy/git subprocess", calls == [], str(calls))
        check("read-only: syntax check still runs", "python syntax" in out and "skipped in read-only" in out)
        D.run_diagnostics.invoke({}, config=_c(root, read_only=False))
        check("writable run still invokes external linters when installed",
              bool(calls) or not (D._which("ruff") or D._which("mypy")), str(calls))
    finally:
        D._run = orig
        shutil.rmtree(root, ignore_errors=True)
    os.environ.update({"MY_CREDENTIAL": "c", "PAT": "p", "DATABASE_URL": "u", "ConnectionStrings__Default": "s",
                       "PLAIN_VAR": "ok"})
    try:
        env = SH._child_env()
        check("broader secret names stripped",
              not any(k in env for k in ("MY_CREDENTIAL", "PAT", "DATABASE_URL", "ConnectionStrings__Default")))
        check("normal vars kept", env.get("PLAIN_VAR") == "ok")
    finally:
        for k in ("MY_CREDENTIAL", "PAT", "DATABASE_URL", "ConnectionStrings__Default", "PLAIN_VAR"):
            os.environ.pop(k, None)


def test_grep_and_read_hardening():
    print("== grep / read_file ==")
    root = _tmp()
    (root / "big.log").write_bytes(b"x" * 10 + b"\n" + (b"ERROR boom\n" * 3) + b"y" * 3_300_000 + b"\n")
    (root / "small.txt").write_text("ERROR small\n", encoding="utf-8")
    out = grep_search.invoke({"pattern": "ERROR", "path": "."}, config=_c(root))
    check("3.3 MB log is searched, not dropped", "big.log:2:" in out and "small.txt:1:" in out, out)
    (root / "u16.log").write_bytes("hello\nERROR utf16 line\n".encode("utf-16"))
    out = grep_search.invoke({"pattern": "ERROR", "path": "u16.log"}, config=_c(root))
    check("UTF-16 (BOM) log is searched", "u16.log:2: ERROR utf16 line" in out, out)
    out = read_file.invoke({"path": "u16.log"}, config=_c(root))
    check("read_file decodes UTF-16", "2: ERROR utf16 line" in out, out)
    (root / "bin.dat").write_bytes(b"ERROR\x00\x01")
    out = grep_search.invoke({"pattern": "ERROR", "path": "."}, config=_c(root))
    check("skipped binary reported even when others match", "not searched: bin.dat" in out, out)
    out = grep_search.invoke({"pattern": "(a+)+$", "path": "."}, config=_c(root))
    check("nested-quantifier regex refused", out.startswith("Error") and "nested quantifier" in out, out)
    (root / ".env").write_text("TOKEN=abc ERROR\n", encoding="utf-8")
    out = read_file.invoke({"path": ".env"}, config=_c(root))
    check("read_file refuses .env", out.startswith("Error") and "abc" not in out, out)
    out = grep_search.invoke({"pattern": "abc", "path": "."}, config=_c(root))
    check("grep does not search .env", "TOKEN" not in out, out)
    shutil.rmtree(root, ignore_errors=True)


def test_symlink_escape_and_move_guards():
    print("== links leaving the root / move_file guards ==")
    root, outside = _tmp(), _tmp()
    (outside / "secret.txt").write_text("TOPSECRET\n", encoding="utf-8")
    try:
        os.symlink(outside / "secret.txt", root / "leak.txt")
        linked = True
    except (OSError, NotImplementedError):
        linked = False
        print("  [SKIP] symlink creation not permitted here")
    if linked:
        out = grep_search.invoke({"pattern": "TOPSECRET", "path": "."}, config=_c(root))
        check("grep ignores a file link leaving the root", "TOPSECRET" not in out, out)
        out = list_directory.invoke({"path": "."}, config=_c(root))
        check("list_directory hides it", "leak.txt" not in out, out)
    if os.name == "nt":  # junctions (no admin needed) are not symlinks to os.walk
        import subprocess
        subprocess.run(["cmd", "/c", "mklink", "/J", str(root / "jn"), str(outside)], capture_output=True)
        if (root / "jn").exists():
            out = grep_search.invoke({"pattern": "TOPSECRET", "path": "."}, config=_c(root))
            check("grep does not follow a junction leaving the root", "TOPSECRET" not in out, out)
            dst = _tmp()
            R.copy_tree_limited(root, dst)
            check("propose copy skips the junction", not (dst / "jn").exists())
            shutil.rmtree(dst, ignore_errors=True)
            subprocess.run(["cmd", "/c", "rmdir", str(root / "jn")], capture_output=True)  # unlink, keep target
    (root / "d").mkdir()
    (root / "d" / "f.txt").write_text("x", encoding="utf-8")
    out = move_file.invoke({"src": "d", "dst": "d/sub"}, config=_c(root))
    check("directory into itself -> Error string", out.startswith("Error"), out)
    out = move_file.invoke({"src": "d", "dst": "e"}, config=_c(root, allow_write=["**"]))
    check("directory move refused under --allow-write", out.startswith("Error") and (root / "d").exists(), out)
    out = move_file.invoke({"src": "d", "dst": "e"}, config=_c(root))
    check("directory move fine without allow-write", out.startswith("OK") and (root / "e" / "f.txt").exists(), out)
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(outside, ignore_errors=True)


def test_zip_bomb_cap():
    print("== zip entry cap ==")
    d = _tmp()
    zp = d / "b.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.txt", b"0" * 5_000_000)
    with zipfile.ZipFile(zp) as zf:
        try:
            S._read_capped(zf, "big.txt", limit=1_000_000)
            ok = False
        except S.EntryTooLarge:
            ok = True
        check("oversized entry refused", ok)
        check("normal read unchanged", len(S._read_capped(zf, "big.txt")) == 5_000_000)
    shutil.rmtree(d, ignore_errors=True)


def test_partial_usage_on_model_failure():
    print("== usage survives a failing model call ==")
    from coding_agent.cli import RunFailed
    from coding_agent import graph as G

    class Boom:
        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, msgs, *a, **kw):
            raise RuntimeError("429 after retries")

    root = _tmp()
    cfg = Config(project_root=root)
    old_build = cli.build_llm
    cli.build_llm = lambda c: Boom()
    try:
        try:
            cli.run_agent(cfg, "task", "run")
            err = None
        except RunFailed as rf:
            err = rf
        except Exception as exc:  # graph may wrap differently; still must not pass silently
            err = exc
        check("failure surfaces as RunFailed", isinstance(err, RunFailed), repr(err))
        if isinstance(err, RunFailed):
            check("step of the failed call is counted", (err.final.get("usage") or {}).get("steps") == 1,
                  str(err.final.get("usage")))
    finally:
        cli.build_llm = old_build
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    for t in (test_main_run_paths, test_execute_run_propose_disables_shell, test_soft_finish_and_shell_warning,
              test_sweep_and_copy_hygiene, test_diagnostics_read_only_and_env, test_grep_and_read_hardening,
              test_symlink_escape_and_move_guards, test_zip_bomb_cap, test_partial_usage_on_model_failure):
        t()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
