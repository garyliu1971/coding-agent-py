"""Command-line interface for the coding agent.

Usage examples::

    python -m coding_agent --root C:\\path\\to\\project analyze --output report.md
    python -m coding_agent --root . run "add a unit test for utils.py"
    python -m coding_agent --root . --read-only chat
    python -m coding_agent --root . run --task-file task.md --json-result r.json --quiet

Exit codes: 0 finished | 2 usage/config error | 3 incomplete (budget, stall,
no final answer) | 4 unexpected failure | 130 interrupted.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import __version__
from . import result as _result
from .config import Config, detect_toolset
from .graph import build_graph
from .llm import build_llm
from .prompts import ANALYZE_TASK, build_system_prompt

console = Console()

# ------------------------------------------------------------------ logging --
_session_log: logging.Logger | None = None
_session_log_path: Path | None = None


class RunFailed(Exception):
    """The agent loop raised; ``final`` is the last state snapshot seen (for usage)."""

    def __init__(self, cause: BaseException, final: dict):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.final = final


def _setup_logging(project_root: Path) -> logging.Logger:
    """Set up a per-session log file under <project_root>/logs/ (or coding-agent/logs/).

    Records:
    - Every tool call (name + truncated args)
    - Every tool result (truncated)
    - LLM token usage (input/output/total) after each LLM turn
    - HTTP 429 / retry warnings from the openai SDK
    - WARNING+ messages from langchain / httpx

    The log file name embeds a Unix timestamp so back-to-back sessions never collide.
    """
    # Write logs next to the coding-agent package, not inside the target project.
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = int(time.time())
    log_path = log_dir / f"session_{ts}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger = logging.getLogger("coding_agent")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Also capture 429/retry noise from the openai SDK and httpx at WARNING+
    for lib in ("openai", "httpx", "langchain", "langgraph"):
        lib_logger = logging.getLogger(lib)
        if not any(isinstance(h, logging.FileHandler) for h in lib_logger.handlers):
            lib_logger.addHandler(fh)
        lib_logger.setLevel(logging.WARNING)

    global _session_log_path
    _session_log_path = log_path
    logger.info("Session started | project=%s | log=%s", project_root, log_path)
    console.print(f"[dim]Session log: {log_path}[/dim]")
    return logger


# ------------------------------------------------------------------ helpers --
def _compact_args(args: dict, limit: int = 200) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > limit:
            s = s[:limit] + "..."
        s = s.replace("\n", "\\n")
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _print_message(m) -> None:
    log = logging.getLogger("coding_agent.messages")
    if isinstance(m, SystemMessage):
        return
    if isinstance(m, HumanMessage):
        return  # the user already saw their own input
    if isinstance(m, AIMessage):
        if m.content:
            console.print(Markdown(str(m.content)))
            log.info("AI: %s", str(m.content)[:500])
        for tc in m.tool_calls or []:
            console.print(
                f"[bold cyan]⚙ {tc.get('name')}([/bold cyan]"
                f"[cyan]{_compact_args(tc.get('args') or {})}[/cyan]"
                f"[bold cyan])[/bold cyan]"
            )
            log.info("TOOL_CALL: %s(%s)", tc.get('name'), _compact_args(tc.get('args') or {}))
        # log token usage if present
        um = getattr(m, "usage_metadata", None)
        if um:
            log.info(
                "USAGE: input=%s output=%s total=%s",
                um.get("input_tokens"), um.get("output_tokens"), um.get("total_tokens"),
            )
    elif isinstance(m, ToolMessage):
        content = str(m.content)
        console.print(
            Panel(
                content[:4000],
                title=f"tool: {getattr(m, 'name', '?')}",
                border_style="dim",
                expand=False,
            )
        )
        log.info("TOOL_RESULT: %s -> %s", getattr(m, 'name', '?'), content[:300])


def _invoke_config(cfg: Config) -> dict:
    return {
        "recursion_limit": cfg.recursion_limit,  # max_iterations*2+20; each hop costs 2 nodes
        "configurable": {
            "project_root": str(cfg.project_root),
            "read_only": cfg.read_only,
            "allow_shell": cfg.allow_shell,
            "allow_write": list(getattr(cfg, "allow_write", None) or []),
            "allow_dangerous_commands": cfg.allow_dangerous_commands,
            "shell_timeout": cfg.shell_timeout,
            "tool_output_limit": cfg.tool_output_limit,
            "file_read_limit": cfg.file_read_limit,
            # Vision / describe_image backend
            "vision_model_url": cfg.vision_model_url,
            "vision_model_name": cfg.vision_model_name,
            "vision_model_api_key": cfg.vision_model_api_key,
            "vision_timeout": cfg.vision_timeout,
            "vision_max_retries": cfg.vision_max_retries,
        },
    }


def run_agent(
    cfg: Config,
    task: str,
    mode: str,
    initial_messages: list | None = None,
    initial_summary: str = "",
    initial_compacted_count: int = 0,
) -> dict:
    """Run the agent loop and stream progress. Returns the final state."""
    graph = build_graph(cfg, build_llm(cfg))
    msgs = list(initial_messages or [])
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs.insert(0, SystemMessage(build_system_prompt(cfg, mode)))
    msgs.append(HumanMessage(task))

    state: dict = {
        "messages": msgs,
        "project_root": str(cfg.project_root),
        "mode": mode,
        "task": task,
        "finished": False,
        "final_summary": "",
        "summary": initial_summary,
        "compacted_count": initial_compacted_count,
    }
    seen = len(msgs)  # don't re-print history we already know about
    final = state
    try:
        for snapshot in graph.stream(state, config=_invoke_config(cfg), stream_mode="values"):
            final = snapshot
            for m in snapshot["messages"][seen:]:
                _print_message(m)
            seen = len(snapshot["messages"])
    except GraphRecursionError:
        console.print(
            "[yellow]⚠ Agent stopped: it hit the maximum number of steps without "
            "finishing. Partial progress is kept.[/yellow]"
        )
        final = {**final, "stop_reason": final.get("stop_reason") or "max_steps"}
    except Exception as exc:  # keep the last snapshot so usage/files survive
        partial = getattr(exc, "partial_usage", None)
        if partial and isinstance(final, dict):
            final = {**final, "usage": partial}
        raise RunFailed(exc, final) from exc
    return final


def _extract_analysis(final: dict) -> str:
    """Return the full analysis report.

    Priority: the <final_analysis> block from the last assistant message first
    (that's the actual report); the `finish` summary second (short fallback);
    then any remaining assistant text.
    """
    last_text = ""
    for m in final.get("messages") or []:
        if isinstance(m, AIMessage) and m.content:
            last_text = str(m.content)
    m = re.search(r"<final_analysis>(.*?)</final_analysis>", last_text, re.S)
    if m:
        return m.group(1).strip()
    if final.get("final_summary"):
        return str(final["final_summary"])
    return last_text.strip()


# ------------------------------------------------------------------- modes --
def _say(text: str) -> None:
    """Plain stdout line that survives --quiet (ASCII-safe)."""
    print(text, flush=True)


def execute_run(cfg: Config, task: str, *, runner=None, propose: bool = False) -> dict:
    """Run ``task`` and return the run-result dict (see result.build_result).

    Agent exceptions are captured as status "failed" rather than raised.  With
    ``propose`` the agent works on a temporary copy of the root and the unified
    diff against the original is returned under "diff"; the original is never
    modified.  Raises result.ProposeTooLarge before running if the copy is too big.
    """
    runner = runner or run_agent
    t0 = time.monotonic()
    tmp: Path | None = None
    run_cfg = cfg
    warnings: list[str] = []
    diff: str | None = None
    final: dict | None = None
    error: str | None = None
    try:
        if propose:
            tmp = Path(tempfile.mkdtemp(prefix="coding-agent-propose-"))
            _result.copy_tree_limited(cfg.project_root, tmp)
            run_cfg = copy.copy(cfg)  # shallow: keeps dynamic attrs such as allow_write
            run_cfg.project_root = tmp
            run_cfg.allow_shell = False  # shell commands could reach the real tree by absolute path
        try:
            final = runner(run_cfg, task, "run")
        except RunFailed as rf:
            final, error = rf.final, str(rf)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if tmp is not None:
            diff, skipped = _result.unified_diff_trees(cfg.project_root, tmp)
            if skipped:
                warnings.append("binary/oversized files changed but not in diff: " + ", ".join(skipped[:10]))
    except BaseException:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp = None
        raise
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    answer = _extract_analysis(final) if final else ""
    return _result.build_result(
        final=final, answer=answer, model=cfg.model, root=str(cfg.project_root),
        read_only=cfg.read_only,
        session_log=str(_session_log_path) if _session_log_path else None,
        wall_seconds=time.monotonic() - t0, warnings=warnings, error=error, diff=diff,
        files_root=run_cfg.project_root,
    )


def _write_outputs(result: dict, output: str | None, json_result: str | None) -> None:
    if output:
        out = _result.write_text_utf8(output, _result.render_output(result))
        if _session_log:
            _session_log.info("run output written to %s", out)
    if json_result:
        _result.write_json_result(json_result, result)


def cmd_analyze(cfg: Config, args) -> int:
    console.print(f"[bold]Analyzing project:[/bold] {cfg.project_root}")
    try:
        final = run_agent(cfg, ANALYZE_TASK, "analyze")
    except RunFailed as rf:
        console.print(f"[red]Analysis failed:[/red] {rf}")
        if args.output:
            _result.write_text_utf8(args.output, f"NO RESULT: run failed ({rf})")
        return _result.EXIT_FAILED
    report = _extract_analysis(final)
    stop = final.get("stop_reason") or "no_final_answer"
    if not report:
        report = f"NO RESULT: stop_reason={stop}"

    if args.output:
        out = _result.write_text_utf8(args.output, report)
        console.print(f"\n[green]✔ Report written to {out}[/green]")
    else:
        console.print()
        console.print(Panel(report, title="Architecture Analysis", border_style="green"))
    return _result.exit_code_for(_result.status_for(stop))


def cmd_run(
    cfg: Config,
    task: str,
    output: str | None = None,
    *,
    json_result: str | None = None,
    quiet: bool = False,
    propose: bool = False,
    runner=None,
) -> int:
    if quiet:
        _say(f"coding-agent: start model={cfg.model} root={cfg.project_root} "
             f"log={_session_log_path or '-'}")
    else:
        console.print(f"[bold]Running task:[/bold] {task}")
    try:
        result = execute_run(cfg, task, runner=runner, propose=propose)
    except _result.ProposeTooLarge as exc:
        result = _result.failure_result(
            str(exc), model=cfg.model, root=str(cfg.project_root), read_only=cfg.read_only,
            session_log=str(_session_log_path) if _session_log_path else None)
        console.print(f"[red]{exc}[/red]")
        if quiet:
            print(f"coding-agent: {exc}", file=sys.stderr, flush=True)
        _write_outputs(result, output, json_result)
        return _result.EXIT_USAGE
    _write_outputs(result, output, json_result)
    line = _result.status_line(result)
    if quiet:
        _say(line)
    else:
        if output:
            console.print(f"\n[green]✔ Output written to {output}[/green]")
        console.print(f"[bold]{line}[/bold]")
        if result["error"]:
            console.print(f"[red]error: {result['error']}[/red]")
    if _session_log:
        _session_log.info("RESULT %s", line)
    return _result.exit_code_for(result["status"])


def cmd_chat(cfg: Config) -> int:
    console.print(
        Panel(
            f"[bold]coding-agent[/bold] · project: [cyan]{cfg.project_root}[/cyan]\n"
            "Type your question. Commands: [cyan]exit[/cyan] / [cyan]quit[/cyan] to leave.",
            border_style="blue",
        )
    )
    # Keep the full message history across turns.
    # We pass it as initial_messages so run_agent prepends the system prompt
    # only on the first call (it checks if msgs[0] is already a SystemMessage).
    # On subsequent calls the history already starts with a SystemMessage so it
    # won't be duplicated — and LangGraph's add_messages reducer appends only
    # the NEW messages from the run, so we replace history with the full final
    # state each time rather than ever appending twice.
    history: list = []
    summary: str = ""
    compacted_count: int = 0
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye 👋")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        final = run_agent(
            cfg,
            user_input,
            "chat",
            initial_messages=history,
            initial_summary=summary,
            initial_compacted_count=compacted_count,
        )
        # Replace history with the full message list from the completed run.
        # Do NOT append — run_agent already returns the complete accumulated state.
        history = final["messages"]
        summary = final.get("summary", "")
        compacted_count = final.get("compacted_count", 0)


# ---------------------------------------------------------------------- cli --
def _nonneg_int(v: str) -> int:
    n = int(v)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def _pos_int(v: str) -> int:
    n = int(v)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return n


_EXIT_HELP = (
    "Exit codes: 0 finished | 2 usage/config error | 3 incomplete "
    "(token/time/step budget, stalled, no final answer) | 4 unexpected failure | 130 interrupted."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="A LangGraph + DeepSeek CLI coding agent for architecture analysis and code modification.",
        epilog=_EXIT_HELP,
    )
    parser.add_argument("--version", action="version", version=f"coding-agent {__version__}")
    parser.add_argument("--root", default=None, help="project root to operate on (default: cwd)")
    parser.add_argument(
        "--model", default=None, help="model id (default: deepseek-chat)"
    )
    parser.add_argument(
        "--api-key", default=None, help="key for the OpenAI-compatible endpoint (local Ollama: no key needed)"
    )
    parser.add_argument("--read-only", action="store_true", help="read-only: no edits, no shell")
    parser.add_argument(
        "--gpt5-mini", action="store_true",
        help="use gpt-5-mini (Azure Foundry) as the main LLM instead of DeepSeek",
    )
    parser.add_argument("--iterations", type=int, default=None, help="max agent loop steps (default 40)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="produce an architecture analysis of the project")
    p_analyze.add_argument("--output", default=None, help="write the report to this file")

    p_run = sub.add_parser("run", help="execute a one-shot task (may modify code)", epilog=_EXIT_HELP)
    p_run.add_argument("task", nargs="*", help="the task description ('-' = read it from stdin)")
    p_run.add_argument(
        "--task-file", default=None, metavar="FILE",
        help="read the task from FILE (UTF-8; '-' = stdin) instead of the command line",
    )
    p_run.add_argument(
        "--output", default=None, metavar="FILE",
        help="write the final agent output to FILE (UTF-8 markdown, never empty); useful for subagent calls to avoid terminal truncation",
    )
    p_run.add_argument(
        "--json-result", default=None, metavar="FILE",
        help="write a machine-readable result (status, stop_reason, answer, files_changed, usage, ...) to FILE as UTF-8 JSON",
    )
    p_run.add_argument(
        "--quiet", "-q", action="store_true",
        help="no live stream; print only a start line and a final 'status=... stop=... tokens=... files=N' line",
    )
    p_run.add_argument("--max-tokens", type=_nonneg_int, default=None, metavar="N",
                       help="token budget for the run (input+output; 0 = unlimited; default 250000)")
    p_run.add_argument("--timeout", type=_pos_int, default=None, metavar="SEC",
                       help="wall-clock budget in seconds (default 600); afterwards the agent is forced to wrap up")
    p_run.add_argument(
        "--allow-write", action="append", default=None, metavar="GLOB",
        help="repeatable; only these root-relative paths (e.g. 'src/**/*.py') may be written/edited/deleted/moved "
             "(env CODING_AGENT_ALLOW_WRITE, ';'-separated); implies --no-shell",
    )
    p_run.add_argument("--no-shell", action="store_true", help="disable run_shell (edit tools stay available)")
    p_run.add_argument(
        "--propose", action="store_true",
        help="work on a temporary copy of the root and report a unified diff; the real project is never modified "
             "(implies --no-shell; refused above 50 MB / 5000 files)",
    )
    p_run.add_argument(
        "--tools", choices=("auto", "all", "core"), default="auto",
        help="optional toolsets: auto = SRDP/vision tools only if the task mentions them; all; core = neither",
    )

    sub.add_parser("chat", help="start an interactive session")

    return parser


def _apply_tools_mode(cfg: Config, mode: str, task: str | None) -> None:
    if mode == "all":
        cfg.enable_srdp = cfg.enable_vision = True
    elif mode == "core":
        cfg.enable_srdp = cfg.enable_vision = False
    elif task is not None:  # auto: only `run` has a task text to inspect
        det = detect_toolset(task)
        cfg.enable_srdp, cfg.enable_vision = det["enable_srdp"], det["enable_vision"]


def _fail_early(args, msg: str, code: int, cfg: Config | None = None) -> int:
    """Report a usage/config error; also honour --json-result/--output if given."""
    console.print(f"[red]{msg}[/red]")
    if console.quiet:  # --quiet silences rich, but errors must still reach the caller
        print(msg, file=sys.stderr, flush=True)
    if getattr(args, "command", None) == "run":
        res = _result.failure_result(
            msg, model=cfg.model if cfg else (args.model or ""),
            root=str(cfg.project_root) if cfg else str(args.root or ""),
            read_only=bool(args.read_only),
            session_log=str(_session_log_path) if _session_log_path else None)
        try:
            _write_outputs(res, getattr(args, "output", None), getattr(args, "json_result", None))
        except OSError:
            pass
    return code


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # rich glyphs must not crash on cp1252 pipes
        try:
            if (stream.encoding or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    is_run = args.command == "run"
    quiet = bool(is_run and args.quiet)
    if quiet:
        console.quiet = True

    task = None
    if is_run:
        try:
            task = _result.resolve_task(args.task, args.task_file)
        except ValueError as exc:
            return _fail_early(args, f"Error: {exc}", _result.EXIT_USAGE)

    cfg = Config.from_env(
        project_root=args.root,
        model=args.model,
        api_key=args.api_key,
        read_only=args.read_only if args.read_only else None,
        max_iterations=args.iterations,
        max_total_tokens=args.max_tokens if is_run else None,
        run_timeout_sec=args.timeout if is_run else None,
    )
    if is_run:
        _apply_tools_mode(cfg, args.tools, task)
        globs = args.allow_write or os.environ.get("CODING_AGENT_ALLOW_WRITE", "").split(";")
        cfg.allow_write = [g.strip() for g in globs if g.strip()]
        # run_shell can write anywhere under (or outside) the root, so neither an
        # --allow-write scope nor --propose is enforceable while it is available.
        if args.no_shell or args.propose or cfg.allow_write:
            cfg.allow_shell = False

    if args.gpt5_mini:
        try:
            cfg.apply_gpt5_mini()
        except ValueError as exc:
            return _fail_early(args, str(exc), _result.EXIT_USAGE, cfg)

    # Set up session log file (must come after cfg is fully built)
    global _session_log
    _session_log = _setup_logging(cfg.project_root)
    try:
        _result.sweep_stale_temp()  # leftovers of killed --propose runs / old edit backups
    except Exception:
        pass
    _session_log.info(
        "Command: %s | model=%s | root=%s | read_only=%s",
        args.command, cfg.model, cfg.project_root, cfg.read_only,
    )

    from .llm import is_local_ollama
    if not cfg.api_key and not is_local_ollama(cfg.base_url):
        return _fail_early(
            args,
            "No API key configured.\n"
            "Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) and DEEPSEEK_BASE_URL in the "
            "environment, or create a .env file:\n"
            "  DEEPSEEK_API_KEY=sk-...\n"
            "  DEEPSEEK_BASE_URL=https://api.deepseek.com/v1\n"
            "  DEEPSEEK_MODEL=deepseek-chat\n"
            f"Model: {cfg.model} @ {cfg.base_url}",
            _result.EXIT_USAGE, cfg,
        )

    if not cfg.project_root.is_dir():
        return _fail_early(args, f"Project root is not a directory: {cfg.project_root}",
                           _result.EXIT_USAGE, cfg)

    try:
        if args.command == "analyze":
            return cmd_analyze(cfg, args)
        if args.command == "run":
            return cmd_run(
                cfg, task, output=args.output, json_result=args.json_result,
                quiet=quiet, propose=args.propose,
            )
        if args.command == "chat":
            return cmd_chat(cfg)
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        return 130
    except Exception as exc:  # unexpected failure after startup
        _session_log.exception("Unexpected failure")
        msg = f"{type(exc).__name__}: {exc}"
        console.print(f"[red]Unexpected failure:[/red] {msg}")
        if quiet:
            _say(f"status=failed stop=error tokens=0 files=0 error={msg}")
        if is_run:
            _fail_early(args, msg, _result.EXIT_FAILED, cfg)
        return _result.EXIT_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
