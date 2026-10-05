"""Graph loop test with a scripted (fake) LLM — no API needed.

Verifies:
1. agent -> tools -> agent -> finish -> finalize -> END routing
2. final_summary is extracted from the `finish` tool call
3. a plain text answer (no tool calls) ends the loop via finalize
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from coding_agent.config import Config
from coding_agent.graph import build_graph

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


class FakeBound:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def invoke(self, messages):
        if self.calls < len(self._script):
            msg = self._script[self.calls]
            self.calls += 1
            return msg
        return AIMessage(content="No more scripted calls.")


class FakeLLM:
    def __init__(self, script):
        self._script = script

    def bind_tools(self, tools, **kwargs):
        return FakeBound(self._script)


class RecordingBound(FakeBound):
    def __init__(self, script):
        super().__init__(script)
        self.inputs: list = []

    def invoke(self, messages):
        self.inputs.append(list(messages))
        return super().invoke(messages)


class RecordingFakeLLM(FakeLLM):
    def bind_tools(self, tools, **kwargs):
        return RecordingBound(self._script)


def tool_call(name: str, args: dict, tid: str) -> AIMessage:
    return AIMessage(
        content=f"calling {name}",
        tool_calls=[{"name": name, "args": args, "id": tid, "type": "tool_call"}],
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ca-graph-"))
    (tmp / "app.py").write_text("print('hi')\n", encoding="utf-8")
    cfg = Config(project_root=tmp, context_budget_chars=100_000)

    print("== scenario: explore then finish ==")
    script = [
        tool_call("list_directory", {"path": ".", "max_depth": 2}, "call_1"),
        tool_call("finish", {"summary": "Project explored; found app.py."}, "call_2"),
    ]
    graph = build_graph(cfg, FakeLLM(script))
    state = {
        "messages": [
            SystemMessage("test prompt"),
            HumanMessage("analyze"),
        ],
        "project_root": str(tmp),
        "mode": "run",
        "task": "analyze",
        "finished": False,
        "final_summary": "",
    }
    config = {
        "recursion_limit": 20,
        "configurable": {
            "project_root": str(tmp),
            "read_only": False,
            "shell_timeout": 30,
            "tool_output_limit": 20000,
            "file_read_limit": 60000,
        },
    }
    final = graph.invoke(state, config=config)
    check("finished flag set", final.get("finished") is True)
    check(
        "final_summary extracted",
        final.get("final_summary") == "Project explored; found app.py.",
        repr(final.get("final_summary")),
    )
    names = [m.name for m in final["messages"] if hasattr(m, "name")]
    check("list_directory executed", "list_directory" in names, str(names))
    check("finish executed", "finish" in names, str(names))

    print("== scenario: plain text answer (chat mode) ==")
    script2 = [AIMessage(content="Here is the answer, no tools needed.")]
    graph2 = build_graph(cfg, FakeLLM(script2))
    final2 = graph2.invoke(state, config=config)
    check("ends via finalize", final2.get("finished") is True)
    check(
        "no tool messages produced",
        not any(isinstance(m, ToolMessage) for m in final2["messages"]),
    )

    print("== scenario: loop guard injects a warning on repeated tool calls ==")
    # Model repeats list_directory 4x (identical) then finishes.
    loop_script = [
        tool_call("list_directory", {"path": ".", "max_depth": 2}, "l1"),
        tool_call("list_directory", {"path": ".", "max_depth": 2}, "l2"),
        tool_call("list_directory", {"path": ".", "max_depth": 2}, "l3"),
        tool_call("list_directory", {"path": ".", "max_depth": 2}, "l4"),
        tool_call("finish", {"summary": "done"}, "f1"),
    ]
    rec_llm = RecordingFakeLLM(loop_script)
    graph3 = build_graph(cfg, rec_llm)
    final3 = graph3.invoke(state, config={**config, "recursion_limit": 30})
    # 5 scripted calls must all run without a recursion error.
    check("repeated calls did not blow recursion", final3.get("finished") is True)
    # Re-run with a bound object that records inputs to verify the warning is injected.
    rec = RecordingFakeLLM(loop_script)
    bound_obj = rec.bind_tools([])
    rec.bind_tools = lambda tools, **kw: bound_obj  # ensure the same instance is reused
    graph4 = build_graph(cfg, rec)
    graph4.invoke(state, config={**config, "recursion_limit": 30})
    saw_warning = any(
        any(isinstance(m, SystemMessage) and "loop guard" in m.content for m in inp)
        for inp in bound_obj.inputs
    )
    check("loop guard warning injected", saw_warning)

    print("== scenario: exploration stall (many list/search, no read) ==")
    # 6 consecutive exploration-only tool calls with VARYING args (identical-call
    # detection alone would not fire), then finish.
    stall_script = [
        tool_call("list_directory", {"path": ".", "max_depth": 3}, "s1"),
        tool_call("file_search", {"glob_pattern": "*.{json,md}"}, "s2"),
        tool_call("list_directory", {"path": ".", "max_depth": 3}, "s3"),
        tool_call("file_search", {"glob_pattern": "**/*.ts"}, "s4"),
        tool_call("list_directory", {"path": ".", "max_depth": 3}, "s5"),
        tool_call("file_search", {"glob_pattern": "*.{yml,yaml}"}, "s6"),
        tool_call("finish", {"summary": "stalled then done"}, "s7"),
    ]
    rec2 = RecordingFakeLLM(stall_script)
    bound2 = rec2.bind_tools([])
    rec2.bind_tools = lambda tools, **kw: bound2
    graph5 = build_graph(cfg, rec2)
    final5 = graph5.invoke(state, config={**config, "recursion_limit": 30})
    check("stall run still finishes", final5.get("finished") is True)
    saw_stall = any(
        any(isinstance(m, SystemMessage) and "STOP exploring" in m.content for m in inp)
        for inp in bound2.inputs
    )
    check("exploration-stall warning injected", saw_stall)
    saw_strip = any(
        any(isinstance(m, SystemMessage) and "REMOVED" in m.content for m in inp)
        for inp in bound2.inputs
    )
    check("exploration tools stripped (hard intervention)", saw_strip)

    print("== scenario: wind-down fires after sustained stall (no progress actions) ==")
    # Wind-down now triggers on stall_count >= 30 consecutive no-progress tool calls.
    # Simulate 31 pure read_file calls (no write/edit/shell) then finish.
    # We patch the threshold to 5 for test speed by subclassing _agent_node.
    # Instead, we directly test _stall_count from graph module.
    from coding_agent.graph import _stall_count
    from langchain_core.messages import AIMessage as AI

    def _tc(name, args, tid):
        return AI(
            content=f"call {name}",
            tool_calls=[{"name": name, "args": args, "id": tid, "type": "tool_call"}],
        )

    stall_msgs = [
        _tc("read_file", {"path": "a.py"}, "r1"),
        _tc("read_file", {"path": "b.py"}, "r2"),
        _tc("grep_search", {"pattern": "foo"}, "r3"),
    ]
    check("stall_count=3 for pure reads", _stall_count(stall_msgs) == 3,
          str(_stall_count(stall_msgs)))

    progress_msgs = stall_msgs + [_tc("edit_file", {"path": "a.py", "old_string": "x", "new_string": "y"}, "e1")]
    check("stall_count=0 after edit_file", _stall_count(progress_msgs) == 0,
          str(_stall_count(progress_msgs)))

    mixed = progress_msgs + [
        _tc("read_file", {"path": "c.py"}, "r4"),
        _tc("read_file", {"path": "d.py"}, "r5"),
    ]
    check("stall_count=2 after two reads post-edit", _stall_count(mixed) == 2,
          str(_stall_count(mixed)))

    print(f"\n== RESULT: {PASS} passed, {FAIL} failed ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
