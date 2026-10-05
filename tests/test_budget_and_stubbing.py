"""Offline tests for the core-loop budgets, old-result stubbing and run result.

Covers: stub_old_tool_results (deterministic, state intact, batched, images kept),
token-budget / step-budget / time-budget wind-down + stop_reason, soft finish on
plain text, read-only mode not stalled by reads, usage accounting (incl.
compaction calls), bounded re-summarisation, system prompt sent once,
detect_toolset, optional toolsets and budget env vars.

Run:  PYTHONPATH=. python tests/test_budget_and_stubbing.py   (no model / network)
"""
from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from coding_agent.compaction import stub_old_tool_results, summarize_prefix
from coding_agent.config import Config, detect_toolset
from coding_agent.graph import build_graph
from coding_agent.prompts import build_system_prompt
from coding_agent.tools import build_tools

USAGE = {"input_tokens": 300, "output_tokens": 100, "total_tokens": 400}


def tc(name: str, args: dict, tid: str, usage: dict | None = USAGE) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tid, "type": "tool_call"}],
        usage_metadata=usage,
    )


def finish_msg(summary: str = "done", tid: str = "fin") -> AIMessage:
    return tc("finish", {"summary": summary}, tid)


class Bound:
    def __init__(self, owner, names):
        self.owner = owner
        self.names = names

    def invoke(self, messages):
        o = self.owner
        o.inputs.append((self.names, list(messages)))
        n = o.bound_calls
        o.bound_calls += 1
        return o.policy(self.names, n)


class FakeLLM:
    """policy(tool_names, call_index) -> AIMessage.  Plain invoke() = summariser."""

    def __init__(self, policy, plain_text="SUM", plain_usage=USAGE):
        self.policy = policy
        self.plain_text = plain_text
        self.plain_usage = plain_usage
        self.inputs: list = []
        self.bound_calls = 0
        self.plain_calls = 0
        self.plain_prompts: list[str] = []

    def bind_tools(self, tools, **kwargs):
        return Bound(self, [t.name for t in tools])

    def invoke(self, messages):
        self.plain_calls += 1
        self.plain_prompts.append(str(messages[-1].content))
        return AIMessage(content=self.plain_text, usage_metadata=self.plain_usage)


def read_step(n: int) -> AIMessage:
    """A read_file call with distinct args each time (so the loop guard stays quiet)."""
    return tc("read_file", {"path": "big.txt", "start_line": n + 1, "end_line": n + 30}, f"r{n}")


def workdir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="ca-budget-"))
    (tmp / "big.txt").write_text("".join(f"line {i} " + "x" * 40 + "\n" for i in range(400)), encoding="utf-8")
    return tmp


def run(cfg: Config, llm, tmp: Path, task: str = "go"):
    graph = build_graph(cfg, llm)
    state = {
        "messages": [SystemMessage("SYS"), HumanMessage(task)],
        "project_root": str(tmp), "mode": "run", "task": task,
        "finished": False, "final_summary": "",
    }
    config = {
        "recursion_limit": cfg.recursion_limit,
        "configurable": {
            "project_root": str(tmp), "read_only": cfg.read_only, "shell_timeout": 30,
            "tool_output_limit": 20000, "file_read_limit": 60000,
        },
    }
    return graph.invoke(state, config=config)


def sys_texts(inputs) -> list[str]:
    return [m.content for _, msgs in inputs for m in msgs if isinstance(m, SystemMessage)]


# ---------------------------------------------------------------------------
# stubbing
# ---------------------------------------------------------------------------

def _rounds(n: int, size: int = 2000) -> list:
    msgs = [SystemMessage("s"), HumanMessage("task")]
    for i in range(n):
        msgs.append(tc("read_file", {"path": "a.py", "start_line": i}, f"t{i}"))
        msgs.append(ToolMessage(content="y" * size, name="read_file", tool_call_id=f"t{i}"))
    return msgs


def _stubbed_ids(view) -> list[str]:
    return [m.tool_call_id for m in view if isinstance(m, ToolMessage) and m.content.startswith("[stub]")]


def test_stub_deterministic_and_state_intact():
    msgs = _rounds(10)
    before = [(type(m), m.content) for m in msgs]
    v1 = stub_old_tool_results(msgs, keep_rounds=4, batch_rounds=4)
    v2 = stub_old_tool_results(msgs, keep_rounds=4, batch_rounds=4)
    assert [m.content for m in v1] == [m.content for m in v2]
    assert before == [(type(m), m.content) for m in msgs], "input messages mutated"
    assert _stubbed_ids(v1) == ["t0", "t1", "t2", "t3"]
    stub = next(m for m in v1 if isinstance(m, ToolMessage) and m.tool_call_id == "t0")
    assert stub.name == "read_file" and stub.tool_call_id == "t0"
    for needle in ("read_file", "path='a.py'", "2000 chars", "re-read if needed"):
        assert needle in stub.content, stub.content
    assert len(v1) == len(msgs)
    # recent rounds untouched
    assert all(m.content == "y" * 2000 for m in v1 if isinstance(m, ToolMessage) and m.tool_call_id >= "t4")


def test_stub_boundary_advances_in_batches():
    stubbed = {n: _stubbed_ids(stub_old_tool_results(_rounds(n), 4, 4)) for n in range(4, 14)}
    assert stubbed[4] == [] and stubbed[7] == []
    # 8..11 rounds -> same 4 stubbed (prefix stays byte-stable), 12 rounds -> 8
    assert stubbed[8] == stubbed[9] == stubbed[10] == stubbed[11] == ["t0", "t1", "t2", "t3"]
    assert len(stubbed[12]) == 8 and len(stubbed[13]) == 8
    v9 = [m.content for m in stub_old_tool_results(_rounds(9), 4, 4)][:10]
    v11 = [m.content for m in stub_old_tool_results(_rounds(11), 4, 4)][:10]
    assert v9 == v11, "prefix changed between batch boundaries"


def test_stub_skips_images_short_and_finish_and_disabled():
    msgs = _rounds(10)
    img = json.dumps({"type": "image_url", "alt": "x", "width": 1, "height": 1, "pad": "z" * 1000})
    msgs[3] = ToolMessage(content=img, name="view_image", tool_call_id="t0")
    msgs[5] = ToolMessage(content="short", name="read_file", tool_call_id="t1")
    msgs[7] = ToolMessage(content="f" * 1000, name="finish", tool_call_id="t2")
    view = stub_old_tool_results(msgs, 4, 4)
    assert view[3].content == img
    assert view[5].content == "short"
    assert view[7].content == "f" * 1000
    assert _stubbed_ids(view) == ["t3"]
    assert [m.content for m in stub_old_tool_results(msgs, 0, 4)] == [m.content for m in msgs]


def test_graph_sends_stubs_but_keeps_state():
    tmp = workdir()
    n_reads = 10

    def policy(names, n):
        return finish_msg() if n >= n_reads else read_step(n)

    llm = FakeLLM(policy)
    cfg = Config(project_root=tmp, read_only=True, stub_after_rounds=2, stub_batch_rounds=2)
    final = run(cfg, llm, tmp)
    last_input = llm.inputs[-1][1]
    stubs = [m for m in last_input if isinstance(m, ToolMessage) and m.content.startswith("[stub]")]
    assert len(stubs) >= 4, len(stubs)
    state_tools = [m for m in final["messages"] if isinstance(m, ToolMessage) and m.name == "read_file"]
    assert state_tools and all(not m.content.startswith("[stub]") for m in state_tools), "state was stubbed"
    assert final["stop_reason"] == "finished"


def test_system_prompt_sent_once():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: finish_msg())
    run(Config(project_root=tmp), llm, tmp)
    first = llm.inputs[0][1]
    assert sum(1 for m in first if isinstance(m, SystemMessage) and m.content == "SYS") == 1


# ---------------------------------------------------------------------------
# budgets / stop_reason / usage
# ---------------------------------------------------------------------------

def test_finish_normally():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: read_step(0) if n == 0 else finish_msg("all good"))
    final = run(Config(project_root=tmp), llm, tmp)
    assert final["stop_reason"] == "finished" and final["final_summary"] == "all good"
    u = final["usage"]
    assert u["steps"] == 2 and u["llm_calls"] == 2
    assert u["input_tokens"] == 600 and u["output_tokens"] == 200 and u["total_tokens"] == 800, u


def test_token_budget_winds_down():
    tmp = workdir()

    def policy(names, n):
        if names == ["finish"]:
            return finish_msg("partial findings")
        return read_step(n)

    llm = FakeLLM(policy)
    cfg = Config(project_root=tmp, read_only=True, max_total_tokens=1000)  # 80% = 800 = after 2 calls
    final = run(cfg, llm, tmp)
    assert final["stop_reason"] == "token_budget", final.get("stop_reason")
    assert final["final_summary"] == "partial findings"
    assert final["usage"]["steps"] == 3 and final["usage"]["total_tokens"] == 1200
    assert llm.inputs[-1][0] == ["finish"]
    assert any("[token budget]" in t for t in sys_texts(llm.inputs[-1:]))
    assert not any("[token budget]" in t for t in sys_texts(llm.inputs[:2])), "no per-turn budget overhead"


def test_token_budget_zero_is_unlimited():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: finish_msg() if n >= 3 else read_step(n))
    final = run(Config(project_root=tmp, read_only=True, max_total_tokens=0), llm, tmp)
    assert final["stop_reason"] == "finished"


def test_step_budget_before_recursion_error():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: finish_msg("cut short") if names == ["finish"] else read_step(n))
    cfg = Config(project_root=tmp, read_only=True, max_iterations=5, max_total_tokens=0)
    final = run(cfg, llm, tmp)  # would raise GraphRecursionError if wind-down were missing
    assert final["stop_reason"] == "max_steps" and final["final_summary"] == "cut short"
    assert final["usage"]["steps"] == 5
    assert llm.inputs[-1][0] == ["finish"]


def test_step_budget_when_model_ignores_finish_only():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: read_step(n))  # never finishes, even when finish-only
    cfg = Config(project_root=tmp, read_only=True, max_iterations=4, max_total_tokens=0)
    final = run(cfg, llm, tmp)
    assert final["stop_reason"] == "max_steps"
    assert final["usage"]["steps"] == 4


def test_time_budget_stop_reason():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: finish_msg("forced") if names == ["finish"] else read_step(n))
    final = run(Config(project_root=tmp, run_timeout_sec=-1, read_only=True), llm, tmp)
    assert final["stop_reason"] == "time_budget" and final["final_summary"] == "forced"


def test_soft_finish_on_plain_text():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: AIMessage(content="Here is my answer.", usage_metadata=USAGE))
    final = run(Config(project_root=tmp), llm, tmp)
    assert final["final_summary"] == "Here is my answer."
    assert final["stop_reason"] == "soft_finished"
    assert final["finished"] is True and final["usage"]["steps"] == 1


def test_read_only_not_stalled_by_reads():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: finish_msg() if n >= 16 else read_step(n))
    cfg = Config(project_root=tmp, read_only=True, stall_threshold=12, max_iterations=40, max_total_tokens=0)
    final = run(cfg, llm, tmp)
    assert final["stop_reason"] == "finished", final["stop_reason"]
    assert not any("[wind-down]" in t for t in sys_texts(llm.inputs))
    assert final["usage"]["steps"] == 17


def test_stall_still_applies_when_writable():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: read_step(n))  # reads forever, ignores finish-only
    cfg = Config(project_root=tmp, read_only=False, stall_threshold=4, max_iterations=40, max_total_tokens=0)
    final = run(cfg, llm, tmp)
    assert final["stop_reason"] == "stalled", final["stop_reason"]
    assert any("[wind-down]" in t for t in sys_texts(llm.inputs))
    llm2 = FakeLLM(lambda names, n: finish_msg("ok") if names == ["finish"] else read_step(n))
    final2 = run(cfg, llm2, tmp)
    assert final2["stop_reason"] == "finished" and final2["final_summary"] == "ok"


# ---------------------------------------------------------------------------
# compaction: bounded summary + usage
# ---------------------------------------------------------------------------

def test_summary_capped_and_prior_merged():
    class Big:
        def __init__(self):
            self.prompts = []

        def invoke(self, messages):
            self.prompts.append(messages[-1].content)
            return AIMessage(content="Z" * 10_000, usage_metadata=USAGE)

    msgs = _rounds(8, size=400)
    llm = Big()
    seen = []
    text, cut = summarize_prefix(
        msgs, llm, keep_recent_chars=500, prior_summary="OLD-SUMMARY-MARKER",
        max_summary_chars=1000, usage_cb=seen.append,
    )
    assert cut > 0 and len(text) <= 1000 + 30, len(text)
    assert "OLD-SUMMARY-MARKER" in llm.prompts[0]
    assert seen == [USAGE]


def test_graph_compaction_bounded_and_counted():
    tmp = workdir()
    llm = FakeLLM(lambda names, n: finish_msg() if n >= 14 else read_step(n), plain_text="Q" * 9000)
    cfg = Config(
        project_root=tmp, read_only=True, context_budget_chars=3000, keep_recent_chars=1500,
        summary_max_chars=800, stub_after_rounds=0, max_total_tokens=0,
    )
    final = run(cfg, llm, tmp)
    assert llm.plain_calls >= 2, "expected repeated compaction"
    assert len(final["summary"]) <= 800 + 30, len(final["summary"])
    assert final["usage"]["llm_calls"] == final["usage"]["steps"] + llm.plain_calls
    assert any("PRIOR SUMMARY" in p for p in llm.plain_prompts[1:]), "prior summary not re-summarised"


# ---------------------------------------------------------------------------
# toolsets / config
# ---------------------------------------------------------------------------

def test_detect_toolset():
    d = detect_toolset
    assert d("summarize app.py and fix the typo") == {"enable_srdp": False, "enable_vision": False}
    assert d("configure the logger") == {"enable_srdp": False, "enable_vision": False}
    assert d("Inspect C:\\bak\\SRDPOutput\\x.zip")["enable_srdp"] is True
    assert d("open the .SRDP package")["enable_srdp"] is True
    assert d("what is in this ZIP?")["enable_srdp"] is False  # needs '.zip' or 'srdp'
    for word in ("image", "PNG", "a.jpg", "photo.JPEG", "screenshot", "Vision model", "picture", "Figure 3"):
        assert d(f"look at the {word} please")["enable_vision"] is True, word
    assert d("")["enable_srdp"] is False and d(None)["enable_vision"] is False


def _names(cfg):
    return {t.name for t in build_tools(cfg)}


def test_optional_toolsets_gate_tools_and_prompt():
    full = _names(Config(vision="on"))
    assert {"srdp_list", "srdp_read", "srdp_grep", "srdp_map_ext_content", "read_image_meta", "view_image"} <= full
    light = _names(Config(vision="on", enable_srdp=False, enable_vision=False))
    assert not (light & {"srdp_list", "srdp_read", "srdp_grep", "srdp_map_ext_content",
                         "read_image_meta", "view_image", "describe_image", "get_omitted_image"})
    assert {"read_file", "finish", "run_diagnostics"} <= light
    assert "SRDP" in build_system_prompt(Config(), "run")
    assert "SRDP" not in build_system_prompt(Config(enable_srdp=False), "run")


def test_env_budget_knobs():
    keys = ["CODING_AGENT_MAX_TOTAL_TOKENS", "CODING_AGENT_MAX_ITERATIONS",
            "CODING_AGENT_RUN_TIMEOUT_SEC", "CODING_AGENT_STUB_AFTER_ROUNDS"]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k, v in zip(keys, ["12345", "7", "99", "2"]):
            os.environ[k] = v
        cfg = Config.from_env()
        assert (cfg.max_total_tokens, cfg.max_iterations, cfg.run_timeout_sec, cfg.stub_after_rounds) == (12345, 7, 99, 2)
        os.environ["CODING_AGENT_MAX_ITERATIONS"] = "junk"
        assert Config.from_env().max_iterations == Config.max_iterations
        assert Config.from_env(max_iterations=3).max_iterations == 3  # CLI override wins
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_defaults():
    c = Config()
    assert c.max_iterations == 40 and c.max_total_tokens == 250_000
    assert c.context_budget_chars == 50_000 and c.keep_recent_chars == 15_000
    assert not hasattr(c, "wind_down_steps")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    print(f"\n== RESULT: {len(tests) - failed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
