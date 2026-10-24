"""Compatibility launcher for plain managed Python, without code instrumentation.

The report argument is retained for existing launchers but is not written.
Successful execution does not certify assertions or task acceptance.
"""

from __future__ import annotations

import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType


def run(script: Path, report: Path, import_directory: Path | None = None) -> None:
    del report
    code = compile(script.read_bytes(), str(script), "exec", dont_inherit=True)
    # Bootstrap trusted runner dependencies before exposing the task's imports.
    # -I still isolates the managed interpreter from host PYTHONPATH/user sites.
    sys.path.insert(0, str(import_directory if import_directory is not None else script.resolve().parent))
    sys.argv = [str(script)]
    module = ModuleType("__main__")
    module.__file__ = str(script)
    module.__loader__ = SourceFileLoader("__main__", str(script))
    module.__dict__.update({"__cached__": None})
    sys.modules["__main__"] = module
    exec(code, module.__dict__)


if __name__ == "__main__":
    run(*(Path(argument) for argument in sys.argv[1:4]))
