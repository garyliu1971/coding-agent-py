"""Central configuration. Values come from environment / .env, overridable via CLI."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# Default vision model (separate from the agent's main LLM).
# Uses the OpenAI Chat Completions format so it works with both
# local Ollama vision models and cloud endpoints (Azure AI Foundry, OpenAI).
# Override via CODING_AGENT_VISION_MODEL_URL / CODING_AGENT_VISION_MODEL_NAME.
DEFAULT_VISION_MODEL_URL = ""          # empty = fall back to DEEPSEEK_BASE_URL
DEFAULT_VISION_MODEL_NAME = "gpt-5-mini"

# Directories that should never be indexed / listed by the exploration tools.
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".next", ".nuxt", "dist", "build", ".idea", ".vscode", ".tox",
    ".mypy_cache", ".pytest_cache", "target", "out", "coverage",
    ".gradle", "bin", "obj", ".cache", ".turbo",
}


def _parse_temperature(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return 0.0
    if raw.strip().lower() in ("none", "default", "omit"):
        return None
    return float(raw)


def _parse_opt_int(v):
    return int(v) if v and v.strip() else None


def _env_int(name: str, default: int) -> int:
    """Int from env var ``name``; blank / invalid values fall back to ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# Keyword triggers for auto-enabling the optional SRDP / vision toolsets.
_SRDP_RE = re.compile(r"srdp|\.zip", re.I)
# Letters-only boundaries so "configure" does not match "figure".
_VISION_RE = re.compile(
    r"(?<![a-z])(?:image|png|jpg|jpeg|screenshot|vision|picture|figure)s?(?![a-z])", re.I
)


def detect_toolset(task_text: str) -> dict:
    """Decide which optional toolsets a task needs, from its text.

    SRDP tools: task mentions srdp / .zip / .srdp.  Vision tools: task mentions
    image, png, jpg, jpeg, screenshot, vision, picture or figure.
    """
    text = task_text or ""
    return {
        "enable_srdp": bool(_SRDP_RE.search(text)),
        "enable_vision": bool(_VISION_RE.search(text)),
    }


@dataclass
class Config:
    """Runtime configuration for the coding agent."""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float | None = 0.0   # None = omit (some GPT-5 deployments reject non-default values)
    max_iterations: int = 40           # max model calls; the last one is finish-only
                                       # (sub-agent default; raise via --iterations / env)
    max_total_tokens: int = 250_000    # input+output tokens per run; 0 = unlimited.
                                       # At >=80% only `finish` is offered.
    stall_threshold: int = 12          # consecutive no-write/shell tool calls -> wind-down
                                       # (ignored in read-only mode)
    run_timeout_sec: int = 600        # wall-clock budget; after this only `finish` is offered
    request_timeout_sec: int = 120    # per model call
    max_completion_tokens: int | None = None  # cap per-call output (incl. reasoning); None = provider default
    reasoning_effort: str | None = None       # gpt-5 family: minimal|low|medium|high; None = provider default
    max_retries: int = 6              # SDK retries 429/5xx with backoff, honoring Retry-After
    context_budget_chars: int = 50_000   # trigger compaction when the LLM view exceeds this
    keep_recent_chars: int = 15_000      # chars of recent context to keep un-summarised
    summary_max_chars: int = 3_000       # hard cap on the (re-summarised) compaction summary
    stub_after_rounds: int = 4           # tool results older than this many assistant rounds
                                         # are replaced by one-line stubs in the LLM view (0 = off)
    stub_batch_rounds: int = 4           # stub boundary advances in batches of this many rounds
                                         # (keeps the prompt-cache prefix stable between batches)

    # Optional toolsets (plus their prompt text). Defaults keep old behaviour;
    # the CLI can switch them off / auto-detect them via detect_toolset().
    enable_srdp: bool = True
    enable_vision: bool = True           # also requires vision != "off" for view/describe tools

    # Vision / image handling (default: off)
    vision: str = "off"  # one of 'off'|'auto'|'on'
    vision_max_side: int = 1568
    vision_max_image_bytes: int = 256 * 1024
    vision_keep_recent: int = 5
    vision_image_token_cost: int = 2048
    # describe_image backend (separate from agent LLM; defaults to gpt-5-mini)
    vision_model_url: str = ""               # empty = inherit DEEPSEEK_BASE_URL
    vision_model_name: str = DEFAULT_VISION_MODEL_NAME
    vision_model_api_key: str = ""           # empty = inherit DEEPSEEK_API_KEY
    vision_timeout: int = 60
    vision_max_retries: int = 3

    read_only: bool = False           # when True: no edits, no shell
    allow_shell: bool = True
    allow_dangerous_commands: bool = False  # allow disaster-level shell commands (rm -rf /, format, shutdown, sudo, ...)
    shell_timeout: int = 120          # seconds
    tool_output_limit: int = 12_000   # cap chars returned by run_shell / grep_search
    file_read_limit: int = 20_000     # cap chars returned by read_file
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Build a Config from environment variables, applying CLI overrides on top."""
        load_dotenv()
        cfg = cls(
            api_key=(
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("OLLAMA_API_KEY")
                or ""
            ),
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
            temperature=_parse_temperature(os.getenv("CODING_AGENT_TEMPERATURE")),
            max_completion_tokens=_parse_opt_int(os.getenv("CODING_AGENT_MAX_COMPLETION_TOKENS")),
            reasoning_effort=(os.getenv("CODING_AGENT_REASONING_EFFORT") or None),
            max_total_tokens=_env_int("CODING_AGENT_MAX_TOTAL_TOKENS", cls.max_total_tokens),
            max_iterations=_env_int("CODING_AGENT_MAX_ITERATIONS", cls.max_iterations),
            run_timeout_sec=_env_int("CODING_AGENT_RUN_TIMEOUT_SEC", cls.run_timeout_sec),
            stub_after_rounds=_env_int("CODING_AGENT_STUB_AFTER_ROUNDS", cls.stub_after_rounds),
            stall_threshold=_env_int("CODING_AGENT_STALL_THRESHOLD", cls.stall_threshold),
            read_only=os.getenv("CODING_AGENT_READ_ONLY", "0").lower() in ("1", "true", "yes"),
            allow_dangerous_commands=os.getenv("CODING_AGENT_ALLOW_DANGEROUS", "0").lower() in ("1", "true", "yes"),
            project_root=Path(os.getenv("CODING_AGENT_ROOT", Path.cwd())),
            vision=os.getenv("CODING_AGENT_VISION", "off"),
            vision_max_side=int(os.getenv("CODING_AGENT_VISION_MAX_SIDE", str(1568))),
            vision_max_image_bytes=int(os.getenv("CODING_AGENT_VISION_MAX_IMAGE_BYTES", str(256 * 1024))),
            vision_keep_recent=int(os.getenv("CODING_AGENT_VISION_KEEP_RECENT", str(5))),
            vision_image_token_cost=int(os.getenv("CODING_AGENT_VISION_IMAGE_TOKEN_COST", str(2048))),
            vision_model_url=os.getenv("CODING_AGENT_VISION_MODEL_URL", ""),
            vision_model_name=os.getenv("CODING_AGENT_VISION_MODEL_NAME", DEFAULT_VISION_MODEL_NAME),
            vision_model_api_key=os.getenv("CODING_AGENT_VISION_API_KEY", ""),
            vision_timeout=int(os.getenv("CODING_AGENT_VISION_TIMEOUT", "60")),
            vision_max_retries=int(os.getenv("CODING_AGENT_VISION_MAX_RETRIES", "3")),
        )
        for key, value in overrides.items():
            if value is None or not hasattr(cfg, key):
                continue
            if key == "project_root":
                cfg.project_root = Path(value)
            else:
                setattr(cfg, key, value)
        cfg.project_root = cfg.project_root.expanduser().resolve()
        return cfg

    @property
    def recursion_limit(self) -> int:
        """LangGraph recursion limit.  Each agent→tools hop costs 2 nodes, so
        we need at least max_iterations * 2, plus slack for finalize / start."""
        return self.max_iterations * 2 + 20

    def require_writable(self) -> None:
        if self.read_only:
            raise PermissionError(
                "This action is disabled: the agent is running in read-only mode."
            )

    def apply_gpt5_mini(self) -> "Config":
        """Switch the main LLM to gpt-5-mini (Azure Foundry), reusing vision config.

        The gpt-5-mini deployment lives on the same Azure endpoint / key as the
        ``describe_image`` vision backend, so we reuse
        ``CODING_AGENT_VISION_MODEL_URL`` / ``_NAME`` / ``_API_KEY``.

        gpt-5-* models reject ``temperature=0``, so temperature is omitted.
        """
        if not self.vision_model_url:
            raise ValueError(
                "gpt-5-mini as the main LLM requires "
                "CODING_AGENT_VISION_MODEL_URL (Azure Foundry OpenAI-compatible "
                "endpoint) to be set in .env"
            )
        self.base_url = self.vision_model_url
        self.model = self.vision_model_name or DEFAULT_VISION_MODEL_NAME
        self.api_key = self.vision_model_api_key
        self.temperature = None  # gpt-5 family rejects temperature=0
        return self
