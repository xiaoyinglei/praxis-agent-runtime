"""Observe managed Python assertions; this is execution scope, not task acceptance.

Invoked in the existing restricted subprocess. The report is a separate channel,
never parsed from user stdout. Original assertions evaluate once with Python's
normal optimization behavior; only reached, successful assertions are counted.
"""

from __future__ import annotations

import ast
import atexit
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from uuid import uuid4


def run(script: Path, report: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    marker = "_praxis_assert_" + uuid4().hex
    error_marker = marker + "_error"
    counts = {"assertions_passed": 0, "assertions_failed": 0}

    def observe(passed: bool) -> None:
        key = "assertions_passed" if passed else "assertions_failed"
        counts[key] += 1

    def publish() -> None:
        report.write_text(json.dumps(counts), encoding="utf-8")

    # Registered before task callbacks: LIFO publishes after they finish and
    # includes assertions they execute. This is diagnostic execution evidence,
    # not a security boundary against hostile code in the same interpreter.
    atexit.register(publish)

    class Assertions(ast.NodeTransformer):
        def visit_Assert(self, node: ast.Assert) -> ast.If:
            def counted(value: bool) -> ast.Expr:
                return ast.Expr(ast.Call(ast.Name(marker, ast.Load()), [ast.Constant(value)], []))

            body = ast.Try(
                body=[node, counted(True)],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(error_marker, ast.Load()), name=None, body=[counted(False), ast.Raise()]
                    )
                ],
                orelse=[],
                finalbody=[],
            )
            return ast.copy_location(ast.If(test=ast.Name("__debug__", ast.Load()), body=[body], orelse=[]), node)

    code = compile(ast.fix_missing_locations(Assertions().visit(tree)), str(script), "exec")
    sys.argv = [str(script)]
    module = ModuleType("__main__")
    module.__file__ = str(script)
    module.__loader__ = SourceFileLoader("__main__", str(script))
    module.__dict__.update({"__cached__": None, marker: observe, error_marker: AssertionError})
    sys.modules["__main__"] = module
    exec(code, module.__dict__)


if __name__ == "__main__":
    run(*(Path(argument) for argument in sys.argv[1:3]))
