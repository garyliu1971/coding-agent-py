"""Regression tests for log-analysis behaviour (offline; one optional live test).

1. Line numbers reported by grep_search/read_file match standard tools even when
   the log contains bare CR and form-feed characters (Jenkins progress output).
2. The wall-clock budget forces the agent to a finish-only model.
3. Live (opt-in): the Gatekeeper QA-deploy log must yield the correct policy.
   Run with:  CODING_AGENT_LIVE=1 CODING_AGENT_GK_LOG=C:\\out\\<log>.txt
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from coding_agent.config import Config
from coding_agent.graph import build_graph
from coding_agent.tools import grep_search, read_file

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


def tcfg(root: Path) -> dict:
    return {
        "configurable": {
            "project_root": str(root),
            "read_only": False,
            "shell_timeout": 30,
            "tool_output_limit": 20000,
            "file_read_limit": 60000,
        }
    }


def test_line_numbers() -> None:
    print("== line numbers with CR / form-feed noise ==")
    tmp = Path(tempfile.mkdtemp(prefix="ca-reg-"))
    body = b"line1\r\nprogress 10%\rprogress 50%\rprogress 100%\r\nline2\x0cpage\nTARGET here\r\nlast\n"
    (tmp / "build.log").write_bytes(body)
    expected = body.replace(b"\r\n", b"\n").split(b"\n").index(b"TARGET here") + 1
    out = grep_search.invoke({"pattern": "TARGET", "path": "build.log"}, config=tcfg(tmp))
    check("grep line number matches \\n-based count", f"build.log:{expected}:" in out, out)
    out = read_file.invoke({"path": "build.log", "start_line": expected, "end_line": expected}, config=tcfg(tmp))
    check("read_file shows same line at same number", f"{expected}: TARGET here" in out, out)


class _Bound:
    def __init__(self, names):
        self.names = names

    def invoke(self, messages):
        if self.names == ["finish"]:
            return AIMessage(content="", tool_calls=[{"name": "finish", "args": {"summary": "forced"}, "id": "f1", "type": "tool_call"}])
        return AIMessage(content="", tool_calls=[{"name": "grep_search", "args": {"pattern": "x", "path": "."}, "id": "g1", "type": "tool_call"}])


class _LLM:
    def bind_tools(self, tools, **kwargs):
        return _Bound([t.name for t in tools])

    def invoke(self, messages):
        return AIMessage(content="summary")


def test_time_budget() -> None:
    print("== wall-clock budget forces finish ==")
    tmp = Path(tempfile.mkdtemp(prefix="ca-reg-"))
    (tmp / "a.txt").write_text("hello\n", encoding="utf-8")
    cfg = Config(project_root=tmp, run_timeout_sec=-1, read_only=True)
    graph = build_graph(cfg, _LLM())
    state = {
        "messages": [SystemMessage("p"), HumanMessage("go")],
        "project_root": str(tmp), "mode": "run", "task": "go",
        "finished": False, "final_summary": "",
    }
    final = graph.invoke(state, config={"recursion_limit": 20, **tcfg(tmp)})
    check("finished via finish-only model", final.get("finished") is True and final.get("final_summary") == "forced", str(final.get("final_summary")))
    check("stop_reason is time_budget", final.get("stop_reason") == "time_budget", str(final.get("stop_reason")))


def test_live_gatekeeper() -> None:
    log = os.getenv("CODING_AGENT_GK_LOG", "")
    if os.getenv("CODING_AGENT_LIVE") != "1" or not log or not Path(log).is_file():
        print("== live gatekeeper: SKIPPED (set CODING_AGENT_LIVE=1 and CODING_AGENT_GK_LOG) ==")
        return
    print("== live gatekeeper analysis ==")
    from coding_agent.prompts import build_system_prompt  # noqa: WPS433

    log_path = Path(log)
    cfg = Config.from_env(project_root=str(log_path.parent), read_only=True, run_timeout_sec=240)
    graph = build_graph(cfg)
    task = (
        f"Analyze the log file {log_path.name}. It is a failed QA deployment. Find which Gatekeeper policy "
        "blocked it. Report the rejected resource, original error text, root cause, fix, with log line numbers."
    )
    state = {
        "messages": [SystemMessage(build_system_prompt(cfg, "run")), HumanMessage(task)],
        "project_root": str(cfg.project_root), "mode": "run", "task": task,
        "finished": False, "final_summary": "",
    }
    final = graph.invoke(state, config={"recursion_limit": cfg.recursion_limit, "configurable": {
        "project_root": str(cfg.project_root), "read_only": True, "shell_timeout": 30,
        "tool_output_limit": 20000, "file_read_limit": 60000}})
    s = final.get("final_summary", "")
    check("finished", final.get("finished") is True)
    check("names policy", "linthorizontalpodautoscaler" in s, s[:200])
    check("names resource", "livedoc-tenant-setup" in s, s[:200])
    check("names minReplicas==maxReplicas", "minReplicas" in s and "maxReplicas" in s, s[:200])


def main() -> int:
    test_line_numbers()
    test_time_budget()
    test_live_gatekeeper()
    print(f"\n== RESULT: {PASS} passed, {FAIL} failed ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
