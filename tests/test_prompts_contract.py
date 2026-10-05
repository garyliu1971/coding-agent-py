"""Prompt contract tests (offline): required rules present, size stays bounded.

Run:  PYTHONPATH=. python tests/test_prompts_contract.py
"""
from __future__ import annotations

import traceback

from coding_agent.config import Config
from coding_agent.prompts import build_system_prompt

# Every character is paid for on every model call - keep the prompt lean.
MAX_RUN_CHARS_NO_SRDP = 2500
MAX_RUN_CHARS_WITH_SRDP = 3400
MAX_ANY_MODE_CHARS = 4000


def test_run_prompt_has_report_contract():
    p = build_system_prompt(Config(), "run")
    for needle in ("FINAL REPORT", "STATUS:", "CHANGED:", "VERIFIED:", "not verified", "UNCERTAIN:"):
        assert needle in p, needle


def test_scope_epistemic_and_verification_rules():
    p = build_system_prompt(Config(), "run")
    assert "SCOPE" in p and "only what the task asks" in p
    assert "only know what you have read" in p and "unverified from excerpt" in p
    assert "narrowest check" in p
    assert "re-read" in p


def test_read_only_prompt_has_no_edit_verification_text():
    p = build_system_prompt(Config(read_only=True), "run")
    assert "READ-ONLY MODE" in p and "narrowest check" not in p


def test_analyze_keeps_final_analysis_format():
    p = build_system_prompt(Config(), "analyze")
    assert "<final_analysis>" in p and "FINAL REPORT" not in p


def test_srdp_section_is_optional():
    assert "SRDP" in build_system_prompt(Config(enable_srdp=True), "run")
    assert "SRDP" not in build_system_prompt(Config(enable_srdp=False), "run")


def test_prompt_size_limits():
    assert len(build_system_prompt(Config(enable_srdp=False), "run")) <= MAX_RUN_CHARS_NO_SRDP
    assert len(build_system_prompt(Config(), "run")) <= MAX_RUN_CHARS_WITH_SRDP
    for mode in ("run", "analyze", "chat"):
        for ro in (False, True):
            assert len(build_system_prompt(Config(read_only=ro), mode)) <= MAX_ANY_MODE_CHARS, (mode, ro)


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
