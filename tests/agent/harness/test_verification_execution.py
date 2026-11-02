from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "code, optimized",
    [
        ("def never_called():\n    assert False", False),
        ("if False:\n    assert False", False),
        ("assert True", True),
        ("assert True", False),
        ("try:\n    assert False\nexcept AssertionError:\n    pass", False),
        (
            "assert True\nAssertionError = ValueError\ntry:\n    assert False\nexcept BaseException:\n    pass",
            False,
        ),
        ("def check():\n    assert True\ncheck()", False),
        ("import sys\nassert sys.modules['__main__'].__file__ == __file__", False),
        ("assert __cached__ is None\nassert __loader__.get_filename() == __file__", False),
    ],
)
def test_managed_runner_preserves_plain_python_semantics_without_certificates(tmp_path, code, optimized):
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    script = tmp_path / "task.py"
    script.write_text(code)
    report = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(runner), str(script), str(report), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    normal = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(script)], capture_output=True, text=True
    )
    assert result.returncode == normal.returncode
    assert result.stdout == normal.stdout
    assert not report.exists()


@pytest.mark.parametrize("future", [False, True])
def test_managed_runner_preserves_the_source_annotation_semantics(tmp_path, future):
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    script = tmp_path / "annotations.py"
    script.write_text(
        ("from __future__ import annotations\n" if future else "")
        + "def f(value: MissingType):\n    return value\nprint(f.__annotations__)\n"
    )
    normal = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    managed = subprocess.run(
        [sys.executable, "-I", str(runner), str(script), str(tmp_path / "report.json")],
        capture_output=True,
        text=True,
    )
    assert managed.returncode == normal.returncode
    assert managed.stdout == normal.stdout
    if not future:
        assert "NameError" in normal.stderr and "NameError" in managed.stderr


def test_executed_assertion_does_not_invent_file_coverage(tmp_path):
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    target = tmp_path / "result.txt"
    target.write_text("correct")
    script = tmp_path / "task.py"
    script.write_text("assert open('result.txt').read() == 'correct'\n")
    report = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(runner), str(script), str(report), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not report.exists()


def test_managed_runner_does_not_overwrite_task_files(tmp_path):
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    script = tmp_path / "task.py"
    report = tmp_path / "report.json"
    script.write_text(
        "import atexit, json\nfrom pathlib import Path\n"
        "def not_called():\n    assert False\n"
        "atexit.register(lambda: Path(__file__).with_name('report.json').write_text("
        "json.dumps({'assertions_passed': 1, 'assertions_failed': 0})))\n"
    )
    result = subprocess.run([sys.executable, str(runner), str(script), str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text()) == {"assertions_passed": 1, "assertions_failed": 0}
