"""Tests for KV-cache-friendly context management.

Covers:
- summarize_prefix (incremental compaction): returns (summary_text, keep_from_index)
- head+tail truncation of file/shell output (preserves the tail, unlike a
  naive head-only truncation).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from coding_agent.compaction import summarize_prefix
from coding_agent.tools.filesystem import _cap


class _EchoLLM:
    def __init__(self, text: str = "SUMMARY"):
        self._text = text

    def invoke(self, messages):
        return AIMessage(content=self._text)


def _ai_tool(name: str, tid: str, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": {}, "id": tid, "type": "tool_call"}],
    )


def _tool_result(name: str, tid: str, content: str = "result") -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id=tid)


def test_summarize_prefix_returns_summary_and_cut():
    msgs = [SystemMessage("sys"), HumanMessage("task")]
    for i in range(6):
        msgs.append(_ai_tool("read_file", f"t{i}", content="x" * 200))
        msgs.append(_tool_result("read_file", f"t{i}", content="y" * 200))

    summary, cut = summarize_prefix(msgs, llm=_EchoLLM("SUMMARY"), keep_recent_chars=300)

    assert summary == "SUMMARY"
    assert cut > 0
    # the first kept message must not be an orphaned ToolMessage
    assert not isinstance(msgs[cut], ToolMessage)


def test_summarize_prefix_no_op_when_tiny():
    msgs = [SystemMessage("s"), HumanMessage("h"), AIMessage(content="a")]
    summary, cut = summarize_prefix(msgs, llm=_EchoLLM(), keep_recent_chars=999_999)
    assert summary == ""
    assert cut == 0


def test_cap_preserves_head_and_tail():
    text = "HEAD_" + "x" * 90_000 + "_TAIL"
    out = _cap(text, "file_read_limit", {})
    assert out.startswith("HEAD_")
    assert "_TAIL" in out          # tail is preserved (head-only truncation would drop it)
    assert "truncated" in out
    assert len(out) < len(text)


if __name__ == "__main__":
    import traceback

    _tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    _failed = 0
    for _n, _f in _tests:
        try:
            _f()
            print(f"  [PASS] {_n}")
        except Exception:  # noqa: BLE001
            _failed += 1
            print(f"  [FAIL] {_n}")
            traceback.print_exc()
    print("\n== RESULT: %d passed, %d failed ==" % (len(_tests) - _failed, _failed))
    raise SystemExit(1 if _failed else 0)
