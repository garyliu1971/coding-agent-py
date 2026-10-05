"""Shell execution tool."""
from __future__ import annotations

import locale
import os
import re
import signal
import subprocess
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _cfg, _root, _within


# ------------------------------------------------------------------ guard ---
# Disaster-level command patterns. Refused by default because they can wipe
# whole disks/system state, take the machine down, or escalate privileges.
# Normal project-scoped commands (e.g. `rm -rf ./build`) are NOT blocked.
# Disable this guard with CODING_AGENT_ALLOW_DANGEROUS=1.
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # POSIX: recursive delete of filesystem roots / home / system dirs
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+(?:--no-preserve-root\s+)?/(?:\s|$)", "recursive delete of '/'"),
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+(?:--no-preserve-root\s+)?~(?:\s|/|$)", "recursive delete of home '~'"),
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+(?:--no-preserve-root\s+)?\$(?:HOME|PWD)(?:\s|/|$)", "recursive delete of $HOME/$PWD"),
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+/(?:home|usr|etc|var|bin|sbin|root|boot|lib)(?:\s|$)", "recursive delete of a system directory"),
    # raw device writes / formatting
    (r"\bdd\s+.*\bof=/dev/", "raw write to a /dev/ device"),
    (r"\bmkfs(?:\.\w+)?\b", "filesystem formatting (mkfs)"),
    # Windows destructive commands
    (r"\bformat\s+[A-Za-z]:", "formatting a Windows drive"),
    (r"\b(?:del|erase|rd)\b[^\n]*/[sS][^\n]*[A-Za-z]:\\(?:\s|$|\*)", "recursive delete of a Windows drive root"),
    (r"\bRemove-Item\b[^\n]*-Recurse[^\n]*[A-Za-z]:\\(?:\s|$|\*)", "recursive Remove-Item of a Windows drive root"),
    (r"\b(?:Clear-Disk|Format-Volume)\b", "disk wipe / format (PowerShell)"),
    # shutdown / reboot
    (r"\b(?:shutdown|reboot|halt|poweroff)\b", "shutdown / reboot"),
    (r"\b(?:Stop-Computer|Restart-Computer|shutdown\.exe)\b", "shutdown / reboot (Windows)"),
    # privilege escalation
    (r"\bsudo\b", "privilege escalation (sudo)"),
    # wildcard / system-directory recursive deletes
    (r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+(?:--no-preserve-root\s+)?/\*", "recursive delete of '/*'"),
    (r"\b(?:Remove-Item|rm|ri)\b[^\n]*-r(?:ecurse)?\b[^\n]*[A-Za-z]:\\(?:Windows|Users|Program Files(?: \(x86\))?|ProgramData)(?:\\\*?)?[\"']?(?:\s|$)", "recursive delete of a Windows system/users directory"),
    (r"\b(?:rmdir|rd|del|erase)\b[^\n]*/[sS]\b[^\n]*[A-Za-z]:\\(?:Windows|Users|Program Files(?: \(x86\))?|ProgramData)(?:\\\*?)?[\"']?(?:\s|$)", "recursive delete of a Windows system/users directory"),
    # discards ALL untracked/ignored files in the repo
    (r"\bgit\s+clean\b[^\n]*\s-[a-z]*f", "git clean -f (deletes untracked files)"),
    # fork bomb
    (r":\s*\(\)\s*\{\s*:\s*\|:\s*&\s*\}\s*;?\s*:", "fork bomb"),
    # remote script piped into a shell
    (r"\b(?:curl|wget)\b[^|;\n]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b", "piping a remote script into a shell"),
]

_DANGEROUS_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _DANGEROUS_PATTERNS]


def _dangerous_reason(command: str) -> str | None:
    """Return a human-readable reason if ``command`` looks dangerous, else None."""
    for pattern, label in _DANGEROUS_RE:
        if pattern.search(command):
            return label
    return None


_MAX_TIMEOUT = 600

# Make Windows PowerShell 5.1 emit UTF-8 (default is the OEM code page).
_PS_UTF8_PREFIX = (
    "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
    "$OutputEncoding=[Console]::OutputEncoding;"
)

_SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH",
                      "COOKIE", "CONNECTIONSTRING", "PRIVATE")
_SECRET_NAMES = {"PAT", "DATABASE_URL", "SAS_URL"}
_SECRET_PREFIXES = ("AZURE_", "DEEPSEEK_", "OPENAI_", "ANTHROPIC_", "GITHUB_", "GH_", "CODING_AGENT_")


def _child_env() -> dict[str, str]:
    """Environment for the child shell with obvious secrets removed."""
    env = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if ku in _SECRET_NAMES or ku.startswith(_SECRET_PREFIXES) or any(part in ku for part in _SECRET_NAME_PARTS):
            continue
        env[k] = v
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and all of its descendants."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=15,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _decode(data: Optional[bytes]) -> str:
    """Decode child output: UTF-8 first, then the locale code page (lossy only here)."""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False) or "utf-8", errors="replace")


def _cap_output(out: str, limit: int) -> str:
    """Head+tail truncation on whole lines, stating what was kept."""
    if len(out) <= limit:
        return out
    lines = out.split("\n")
    head_budget = int(limit * 0.7)
    tail_budget = limit - head_budget
    head: list[str] = []
    used = 0
    for ln in lines:
        if used + len(ln) + 1 > head_budget:
            break
        head.append(ln)
        used += len(ln) + 1
    tail: list[str] = []
    used = 0
    for ln in reversed(lines[len(head):]):
        if used + len(ln) + 1 > tail_budget:
            break
        tail.append(ln)
        used += len(ln) + 1
    tail.reverse()
    if not head and not tail:  # one giant line: fall back to a char cut
        return out[:head_budget] + f"\n...[truncated, {len(out) - limit:,} chars omitted]...\n" + out[-tail_budget:]
    skipped = len(lines) - len(head) - len(tail)
    marker = (
        f"...[output truncated: kept first {len(head)} and last {len(tail)} of {len(lines)} lines; "
        f"{skipped} lines / {len(out) - sum(map(len, head)) - sum(map(len, tail)):,} chars omitted]..."
    )
    return "\n".join(head + [marker] + tail)


@tool
def run_shell(
    command: str,
    workdir: str = ".",
    timeout: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Run a shell command in `workdir` (relative to project root). Uses PowerShell on Windows, bash otherwise. Returns stdout, stderr and the exit code. Prefer this over guessing at build/test outputs."""
    if _cfg(config).get("read_only"):
        return "Error: read-only mode; run_shell is disabled."
    if not _cfg(config).get("allow_dangerous_commands", False):
        reason = _dangerous_reason(command)
        if reason:
            return (
                f"Error: refusing to run this command — {reason}.\n"
                "This is a destructive / high-risk operation. If you are certain, "
                "the user must set CODING_AGENT_ALLOW_DANGEROUS=1 and re-run."
            )
    root = _root(config)
    cwd = (root / workdir).resolve()
    if not _within(root, cwd):
        return f"Error: workdir escapes project root: {workdir}"
    if not cwd.is_dir():
        return f"Error: workdir does not exist: {workdir}"
    t = timeout if timeout and timeout > 0 else int(_cfg(config).get("shell_timeout", 120))
    t = min(t, _MAX_TIMEOUT)
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_UTF8_PREFIX + command]
        popen_kw: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        cmd = ["bash", "-lc", command]
        popen_kw = {"start_new_session": True}
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_child_env(),
            **popen_kw,
        )
    except OSError as e:
        return f"Error: could not start shell: {e}"
    timed_out = False
    try:
        out_b, err_b = proc.communicate(timeout=t)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        _kill_tree(proc)
        out_b, err_b = exc.stdout, exc.stderr
        try:  # collect whatever the dead tree left in the pipes; never wait long
            out_b, err_b = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # a detached grandchild still holds the pipes; keep what we have
    stdout, stderr = _decode(out_b), _decode(err_b)
    parts = []
    if stdout:
        parts.append(stdout.strip())
    if stderr:
        parts.append("[stderr]\n" + stderr.strip())
    out = _cap_output("\n".join(parts), int(_cfg(config).get("tool_output_limit", 12_000)))
    if timed_out:
        return f"Error: command timed out after {t}s (process tree killed):\n$ {command}" + (
            f"\n[partial output]\n{out}" if out else ""
        )
    if not out:
        return f"$ {command}\n(exit code {proc.returncode}, no output)"
    return f"$ {command}\n(exit code {proc.returncode})\n{out}"
