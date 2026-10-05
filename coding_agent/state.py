"""Agent state shared across the LangGraph nodes.

Run result contract
-------------------
The FINAL graph state (the last ``values`` snapshot / ``graph.invoke`` result)
exposes the outcome of a run:

* ``final_summary`` (str): the final answer.  The ``summary`` argument of the
  ``finish`` tool call; if the model ended with plain text instead, that text
  (soft finish); "" if there is none.
* ``stop_reason`` (str), one of:
    "finished"         the model called ``finish`` normally
    "token_budget"     cfg.max_total_tokens reached 80%; finish-only wind-down
    "time_budget"      cfg.run_timeout_sec exceeded; finish-only wind-down
    "max_steps"        cfg.max_iterations reached; the last model call is finish-only
    "stalled"          stall wind-down was forced and the model still did not finish
    "soft_finished"    the model ended a turn with a plain-text answer and no ``finish``
                       (counts as success: status ok)
    "no_final_answer"  the run ended with neither a ``finish`` call nor any answer text
  For the budget reasons, ``final_summary`` still holds whatever the model
  reported (finish summary or text) - possibly "".
* ``usage`` (dict): {"input_tokens", "output_tokens", "total_tokens",
  "llm_calls", "steps"} (all int) accumulated over the whole run from the
  model's ``usage_metadata``, including summarisation (compaction) calls.
  ``steps`` counts agent-node model calls only.  It is also present in
  intermediate ``values`` snapshots, so a caller whose run is aborted by an
  exception (e.g. GraphRecursionError, which the step budget makes unlikely)
  can still read usage from the last snapshot; ``stop_reason`` is only set by
  the ``finalize`` node, so it is absent in that case.
* ``finished`` (bool): True once ``finalize`` ran (kept for compatibility; it
  does NOT mean the model called ``finish`` - use ``stop_reason``).

``wind_down`` is internal (the reason a finish-only turn was forced, "" if none).
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Conversation history (LangGraph merges new messages via add_messages).
    messages: Annotated[list[AnyMessage], add_messages]
    project_root: str
    mode: str                 # "analyze" | "run" | "chat"
    task: str
    finished: bool
    final_summary: str
    # Run result (see "Run result contract" above).
    stop_reason: str
    usage: dict
    wind_down: str            # internal: "token_budget"|"time_budget"|"max_steps"|"stalled"|""
    # Persistent context compaction (prompt-cache friendly): the summary text
    # covers the first `compacted_count` messages in `messages`.
    summary: str
    compacted_count: int
