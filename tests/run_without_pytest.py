"""Run pytest-style test modules where pytest is NOT installed.

Usage:  python tests/run_without_pytest.py tests/test_edit_file.py [more files...]
        (no args = test_edit_file, test_image_meta, test_describe_image)

Provides a tiny stand-in for the only pytest features those files use:
``pytest.raises(exc, match=...)`` and the ``tmp_path`` / ``monkeypatch`` fixtures.
If real pytest is installed, prefer it.
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import os
import re
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FILES = ["test_edit_file.py", "test_image_meta.py", "test_describe_image.py"]


@contextlib.contextmanager
def _raises(exc, match=None):
    try:
        yield
    except exc as e:
        if match is not None and not re.search(match, str(e)):
            raise AssertionError(f"exception text {str(e)!r} does not match {match!r}") from None
    else:
        raise AssertionError(f"DID NOT RAISE {exc}")


class _Monkeypatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None):
        if isinstance(target, str):  # "pkg.mod.attr" form is not used by our tests
            raise NotImplementedError
        old = getattr(target, name)
        self._undo.append(lambda: setattr(target, name, old))
        setattr(target, name, value)

    def setenv(self, key, value):
        old = os.environ.get(key)
        self._undo.append(lambda: os.environ.pop(key, None) if old is None else os.environ.__setitem__(key, old))
        os.environ[key] = str(value)

    def delenv(self, key, raising=True):
        if key not in os.environ:
            if raising:
                raise KeyError(key)
            return
        old = os.environ.pop(key)
        self._undo.append(lambda: os.environ.__setitem__(key, old))

    def undo(self):
        while self._undo:
            self._undo.pop()()


def _install_stub() -> None:
    mod = types.ModuleType("pytest")
    mod.raises = _raises
    mod.fixture = lambda f=None, **kw: (f if f is not None else (lambda g: g))
    sys.modules["pytest"] = mod


def _run_callable(fn, label):
    sig = inspect.signature(fn)
    kwargs, cleanups, mp = {}, [], None
    for name in sig.parameters:
        if name == "tmp_path":
            d = Path(tempfile.mkdtemp(prefix="nopytest-"))
            kwargs[name] = d
            cleanups.append(lambda d=d: shutil.rmtree(d, ignore_errors=True))
        elif name == "monkeypatch":
            mp = kwargs[name] = _Monkeypatch()
        elif name != "self":
            raise RuntimeError(f"{label}: unsupported fixture {name!r}")
    try:
        fn(**kwargs)
    finally:
        if mp:
            mp.undo()
        for c in cleanups:
            c()


def main(argv: list[str]) -> int:
    try:
        import pytest  # noqa: F401  (real one wins)
    except ModuleNotFoundError:
        _install_stub()
    sys.path.insert(0, str(HERE.parent))
    files = [Path(a) for a in argv] or [HERE / f for f in DEFAULT_FILES]
    passed = failed = 0
    for f in files:
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cases = []
        for name, obj in vars(mod).items():
            if name.startswith("test") and inspect.isfunction(obj):
                cases.append((f"{f.stem}::{name}", obj))
            elif name.startswith("Test") and inspect.isclass(obj):
                inst = obj()
                for mname, m in inspect.getmembers(inst, inspect.ismethod):
                    if mname.startswith("test"):
                        cases.append((f"{f.stem}::{name}::{mname}", m))
        for label, fn in cases:
            try:
                _run_callable(fn, label)
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {label}")
                traceback.print_exc()
    print(f"== RESULT: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
