"""Context compaction: summarise old messages instead of dropping them.

Problem with plain trim_messages(strategy="last"):
  - It can leave orphaned ToolMessages whose AIMessage(tool_calls) was trimmed,
    which causes API errors on Anthropic/OpenAI (tool_call and tool_result must
    be paired).
  - It discards context silently; in 100+ turn tasks the model re-does work it
    already completed.

This module replaces the trim step with an LLM-generated summary:
  1. Find the last SAFE cut point — an index where the message just before it is
     a complete tool-round (AIMessage with tool_calls followed by all its
     ToolMessages), or a plain AIMessage with no tool_calls.  Never cut inside
     a tool call / tool result pair.
  2. Collect messages BEFORE the cut point (excluding SystemMessage) as the
     "to-summarise" span.
  3. Call the LLM once with a compact prompt to produce a plain-text summary.
  4. Return [system_msg, summary_as_SystemMessage, ...recent_messages].

KV-cache note
-------------
``compact()`` returns a full message list (used by tests / one-shot calls).
``summarize_prefix()`` is the incremental variant used by the agent loop: it
returns ``(summary_text, keep_from_index)`` so the caller can PERSIST the
summary in state.  Persisting (rather than re-summarising every turn) keeps the
leading bytes of each LLM request stable, which is what makes provider prompt
caching hit.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def count_chars(messages: Sequence[AnyMessage], image_token_cost: int = 2048) -> int:
    """Count characters in messages but treat image ToolMessages as fixed cost.

    Args:
        messages: sequence of AnyMessage
        image_token_cost: fixed token-equivalent cost to charge per image
    """
    total = 0
    for m in messages:
        # ToolMessage that encodes an image as a JSON block/dict: count as fixed
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, dict):
                if content.get("type") == "image_url":
                    total += image_token_cost
                    continue
            elif isinstance(content, str):
                try:
                    j = json.loads(content)
                    if isinstance(j, dict) and j.get("type") == "image_url":
                        total += image_token_cost
                        continue
                except Exception:
                    pass
        c = m.content
        total += len(c) if isinstance(c, str) else 0
    return total


def _is_complete_round_end(messages: list[AnyMessage], idx: int) -> bool:
    """Return True if messages[idx-1] is a safe cut boundary.

    Safe = the message at idx-1 is either:
      - A ToolMessage that is the LAST in its batch (i.e. messages[idx] does
        NOT start another ToolMessage belonging to the same AIMessage).
      - A plain AIMessage with no tool_calls.

    We also verify that messages[idx] (the first kept message) is a HumanMessage
    or AIMessage, never a ToolMessage — keeping orphan ToolMessages is invalid.
    """
    if idx <= 0 or idx >= len(messages):
        return False
    prev = messages[idx - 1]
    nxt = messages[idx]

    # The first kept message must not be a ToolMessage (would be orphaned)
    if isinstance(nxt, ToolMessage):
        return False

    # If prev is a ToolMessage, it's safe only if all tool_calls of its
    # originating AIMessage have been answered (no more ToolMessages follow
    # before the next AIMessage).
    if isinstance(prev, ToolMessage):
        return True  # nxt is not ToolMessage already checked above

    # If prev is an AIMessage with no tool calls, safe.
    if isinstance(prev, AIMessage) and not prev.tool_calls:
        return True

    return False


def find_cut_index(messages: list[AnyMessage], keep_chars: int) -> int:
    """Return the index of the first message to KEEP (everything before is summarised).

    We walk backwards from the end accumulating chars until we exceed keep_chars,
    then search forward for the nearest safe cut boundary.  Returns 1 (keep
    everything except system) if no safe cut is found.
    """
    # Ignore leading SystemMessages when scanning
    body_start = 0
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            body_start = i + 1
        else:
            break

    body = messages[body_start:]
    if not body:
        return body_start

    # Walk backwards to find the rough cut point by char budget
    accumulated = 0
    raw_cut = len(body)  # index in body[]
    for i in range(len(body) - 1, -1, -1):
        c = body[i].content
        accumulated += len(c) if isinstance(c, str) else 0
        if accumulated >= keep_chars:
            raw_cut = i
            break

    # Search forward from raw_cut for a safe boundary
    for i in range(raw_cut, len(body)):
        abs_i = body_start + i
        if _is_complete_round_end(messages, abs_i):
            return abs_i

    # Search backward as fallback
    for i in range(raw_cut - 1, -1, -1):
        abs_i = body_start + i
        if _is_complete_round_end(messages, abs_i):
            return abs_i

    return body_start  # summarise nothing; keep everything


# ---------------------------------------------------------------------------
# old tool-result stubbing (LLM-only view; graph state is never modified)
# ---------------------------------------------------------------------------

_STUB_SKIP_ARGS = {"content", "new_string", "old_string", "edits"}
_STUB_MIN_CHARS = 300  # results shorter than this are not worth stubbing


def _is_image_tool_message(m: ToolMessage) -> bool:
    c = m.content
    if isinstance(c, dict):
        return c.get("type") == "image_url"
    if isinstance(c, str) and c.startswith("{"):
        try:
            j = json.loads(c)
            return isinstance(j, dict) and j.get("type") == "image_url"
        except Exception:  # noqa: BLE001
            return False
    return False


def _stub_text(m: ToolMessage, call: dict | None) -> str:
    name = getattr(m, "name", None) or (call or {}).get("name") or "tool"
    args = (call or {}).get("args") or {}
    parts = []
    for k in sorted(args):
        if k in _STUB_SKIP_ARGS:
            continue
        parts.append(f"{k}={repr(args[k])[:60]}")
        if len(parts) >= 4:
            break
    return (
        f"[stub] {name}({', '.join(parts)}) -> {len(m.content)} chars omitted; "
        "re-read if needed"
    )


def stub_old_tool_results(
    messages: list[AnyMessage],
    keep_rounds: int = 4,
    batch_rounds: int = 4,
) -> list[AnyMessage]:
    """Return a view of ``messages`` with old ToolMessage contents stubbed.

    A "round" is one AIMessage.  ToolMessages belonging to rounds older than the
    last ``keep_rounds`` are replaced by a deterministic one-line stub (tool name,
    key args, original size, "re-read if needed").

    Cache note: the stub boundary only advances in steps of ``batch_rounds``
    (stubbed rounds = floor((rounds - keep_rounds) / batch_rounds) * batch_rounds),
    so between advances every leading message is byte-identical and the provider
    prompt-cache prefix stays valid; sliding every turn would invalidate it each turn.

    Image-carrying results, short results and the ``finish`` result are left
    alone.  Input messages are never mutated; stubbed ones are new objects.
    """
    if keep_rounds <= 0:
        return list(messages)
    batch = max(1, batch_rounds)
    ai_idx = [i for i, m in enumerate(messages) if isinstance(m, AIMessage)]
    stub_rounds = ((len(ai_idx) - keep_rounds) // batch) * batch
    if stub_rounds <= 0:
        return list(messages)
    boundary = ai_idx[stub_rounds]  # everything before this index is "old"

    calls: dict[str, dict] = {}
    for m in messages[:boundary]:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc.get("id"):
                    calls[tc["id"]] = tc

    out: list[AnyMessage] = []
    for i, m in enumerate(messages):
        if (
            i < boundary
            and isinstance(m, ToolMessage)
            and isinstance(m.content, str)
            and len(m.content) > _STUB_MIN_CHARS
            and getattr(m, "name", None) != "finish"
            and not _is_image_tool_message(m)
        ):
            text = _stub_text(m, calls.get(m.tool_call_id))
            if len(text) < len(m.content):
                m = ToolMessage(content=text, name=getattr(m, "name", None), tool_call_id=m.tool_call_id)
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# serialisation (for the summary prompt)
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text)-limit} chars]"


def _serialise(messages: list[AnyMessage]) -> str:
    """Convert a message list to a readable text transcript for the summary LLM."""
    parts: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        content = m.content if isinstance(m.content, str) else ""
        if isinstance(m, HumanMessage):
            parts.append(f"[User]: {_truncate(content)}")
        elif isinstance(m, AIMessage):
            if content:
                parts.append(f"[Assistant]: {_truncate(content)}")
            for tc in m.tool_calls or []:
                args_str = ", ".join(f"{k}={repr(v)[:120]}" for k, v in (tc.get("args") or {}).items())
                parts.append(f"[Tool call]: {tc.get('name')}({args_str})")
        elif isinstance(m, ToolMessage):
            # If the tool result is an image_url JSON block, show a compact description
            descr = None
            if isinstance(content, str):
                try:
                    j = json.loads(content)
                    if isinstance(j, dict) and j.get("type") == "image_url":
                        alt = j.get("alt") or getattr(m, "name", "image")
                        w = j.get("width")
                        h = j.get("height")
                        size_str = f" {w}x{h}px" if w and h else ""
                        descr = f"[Tool result (image)]: {alt}{size_str}"
                except Exception:
                    descr = None
            if descr:
                parts.append(descr)
            else:
                parts.append(f"[Tool result ({getattr(m, 'name', '?')})]: {_truncate(content, 600)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# internal split / summarise helpers
# ---------------------------------------------------------------------------

def _split_messages(
    messages: list[AnyMessage],
    keep_recent_chars: int,
) -> tuple[list[AnyMessage], list[AnyMessage], list[AnyMessage], int]:
    """Split ``messages`` into (system_msgs, to_summarise, to_keep, cut_abs).

    ``cut_abs`` is the absolute index (into ``messages``) of the first message
    to keep — everything before it (except leading SystemMessages) is
    summarised.
    """
    system_msgs: list[AnyMessage] = []
    body: list[AnyMessage] = []
    in_system = True
    for m in messages:
        if in_system and isinstance(m, SystemMessage):
            system_msgs.append(m)
        else:
            in_system = False
            body.append(m)

    cut = find_cut_index(messages, keep_recent_chars)
    body_cut = cut - len(system_msgs)
    if body_cut < 0:
        body_cut = 0
    to_summarise = body[:body_cut]
    to_keep = body[body_cut:]
    return system_msgs, to_summarise, to_keep, cut


def _replace_old_images(to_keep: list[AnyMessage], keep_recent_images: int) -> list[AnyMessage]:
    """Replace old inline-image ToolMessages with placeholders, keeping only the
    most recent ``keep_recent_images`` to avoid re-sending large base64 blobs.

    Returns a new list (the caller's list is mutated in place as well).
    """
    image_positions: list[int] = []
    for i, m in enumerate(to_keep):
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            try:
                j = json.loads(m.content)
                if isinstance(j, dict) and j.get("type") == "image_url":
                    image_positions.append(i)
            except Exception:
                continue

    if not image_positions:
        return to_keep

    keep_last = set(image_positions[-keep_recent_images:])
    index_path = Path(__file__).parent / ".." / "compacted_images.json"
    try:
        index_path = index_path.resolve()
    except Exception:
        index_path = Path("compacted_images.json")

    try:
        stored = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    except Exception:
        stored = {}

    for idx in image_positions:
        if idx in keep_last:
            continue
        m = to_keep[idx]
        meta = {"alt": None, "width": None, "height": None}
        try:
            j = json.loads(m.content) if isinstance(m.content, str) else (m.content or {})
            if isinstance(j, dict):
                meta["alt"] = j.get("alt")
                meta["width"] = j.get("width")
                meta["height"] = j.get("height")
        except Exception:
            pass
        alt = meta.get("alt") or getattr(m, "name", "image")
        placeholder_text = f"[image omitted: {alt}] (image removed to reduce token usage)"
        to_keep[idx] = SystemMessage(placeholder_text)
        key = f"omitted_{len(stored)+1}"
        stored[key] = {"placeholder": placeholder_text, "meta": meta}

    try:
        index_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    except Exception:
        pass

    return to_keep


DEFAULT_SUMMARY_MAX_CHARS = 3_000

_SUMMARY_PROMPT = textwrap.dedent("""\
    You are summarising a coding-agent conversation for context compression.
    The agent is working on a software project. Below is the conversation history
    to summarise. Write a concise summary covering:

    - The overall task / goal
    - Key findings from code exploration (important files, architecture, patterns)
    - Changes already made (files edited/created, what was changed and why)
    - Any errors encountered and how they were resolved
    - What has been verified (tests passed, build succeeded, etc.)
    - Current status and what remains to do

    Be specific about file paths and code details — the agent will use this
    summary to continue its work without re-reading already-explored files.
    HARD LIMIT: at most ~{max_words} words. {prior_note}
    Do NOT include meta-commentary. Write only the summary content.
    {prior_block}
    CONVERSATION:
    {transcript}
""")


def _clip_summary(text: str, max_chars: int) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + "…[summary truncated]"
    return text


def _report_usage(response, usage_cb) -> None:
    """Report a summarisation call to ``usage_cb(usage_metadata_or_None)``."""
    if usage_cb is None:
        return
    try:
        usage_cb(getattr(response, "usage_metadata", None))
    except Exception:  # noqa: BLE001 - accounting must never break compaction
        pass


def _generate_summary(
    to_summarise: list[AnyMessage],
    llm,
    prior_summary: str = "",
    max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    usage_cb=None,
) -> str:
    """Call the LLM once to summarise ``to_summarise``; fall back gracefully.

    If ``prior_summary`` is given it is folded into the new summary (the caller
    REPLACES the old summary), so the persistent prefix stays bounded.
    """
    transcript = _serialise(to_summarise)
    prior_block = f"\nPRIOR SUMMARY (merge into the new one):\n{prior_summary}\n" if prior_summary else ""
    prompt = _SUMMARY_PROMPT.format(
        transcript=transcript,
        max_words=max(50, (max_chars or DEFAULT_SUMMARY_MAX_CHARS) // 7),
        prior_note="Merge the PRIOR SUMMARY with the conversation into ONE summary." if prior_summary else "",
        prior_block=prior_block,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        _report_usage(response, usage_cb)
        text = response.content if isinstance(response.content, str) else str(response.content)
        return _clip_summary(text, max_chars)
    except Exception as exc:  # noqa: BLE001
        note = (
            f"[Compaction failed: {exc}] "
            f"Earlier conversation ({len(to_summarise)} messages) was summarised "
            "but the summary could not be generated. Continuing from recent context."
        )
        return _clip_summary((prior_summary + "\n\n" if prior_summary else "") + note, max_chars)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def summarize_prefix(
    messages: list[AnyMessage],
    llm,
    keep_recent_chars: int,
    image_token_cost: int = 2048,
    keep_recent_images: int = 5,
    prior_summary: str = "",
    max_summary_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    usage_cb=None,
) -> tuple[str, int]:
    """Summarise the old prefix of ``messages``.

    Returns ``(summary_text, keep_from_index)`` where ``keep_from_index`` is the
    absolute index of the first message NOT covered by the summary.  When there
    is nothing to summarise, returns ``("", 0)``.

    This is the incremental variant used by the agent loop: the caller persists
    ``summary_text`` in state so the leading bytes of subsequent requests stay
    stable (prompt-cache friendly), instead of re-summarising every turn.

    If ``prior_summary`` is passed it is merged into the returned text, so the
    caller should REPLACE (not append to) its stored summary.  The result is
    capped at ``max_summary_chars``.  ``usage_cb`` receives the summary call's
    ``usage_metadata`` (or None) for token accounting.
    """
    _, to_summarise, to_keep, cut = _split_messages(messages, keep_recent_chars)
    if not to_summarise:
        return "", 0

    _replace_old_images(to_keep, keep_recent_images)
    summary_text = _generate_summary(
        to_summarise, llm, prior_summary=prior_summary,
        max_chars=max_summary_chars, usage_cb=usage_cb,
    )
    return summary_text, cut


def compact(
    messages: list[AnyMessage],
    llm,
    total_budget_chars: int,
    keep_recent_chars: int,
    image_token_cost: int = 2048,
    keep_recent_images: int = 5,
) -> list[AnyMessage]:
    """Return a compacted message list safe for LLM API submission.

    If the total char count is within budget, returns messages unchanged.
    Otherwise:
      - Identifies a safe cut point (no orphaned ToolMessages)
      - Summarises everything before the cut with one LLM call
      - Returns [system_msgs..., summary_SystemMessage, recent_msgs...]

    Args:
        messages: Full message history (including SystemMessage at index 0).
        llm: A plain LLM (NOT bind_tools) — used only for the summary call.
        total_budget_chars: Trigger threshold; no compaction if below this.
        keep_recent_chars: How many recent chars to preserve un-summarised.
    """
    if count_chars(messages, image_token_cost=image_token_cost) <= total_budget_chars:
        return messages

    system_msgs, to_summarise, to_keep, _ = _split_messages(messages, keep_recent_chars)
    if not to_summarise:
        return messages

    _replace_old_images(to_keep, keep_recent_images)
    summary_text = _generate_summary(to_summarise, llm)

    summary_msg = SystemMessage(
        content=f"[CONTEXT SUMMARY — earlier conversation compressed]\n\n{summary_text}"
    )
    return [*system_msgs, summary_msg, *to_keep]
