"""阶段1 CI 硬规则:scripts/check_anti_patterns.py 的 praxis 扫描单元测试。

用 tmp_path 构造违规/合规 fixture,逐条验证 scan_praxis_file 的规则判定。
唯一写入点白名单按"相对路径后缀"匹配(如 store/turns.py),
因此 tmp fixture 也能模拟允许写入点,而不依赖真实 praxis 目录位置。
"""

from __future__ import annotations

from pathlib import Path

from scripts.check_anti_patterns import scan_praxis_file


def _write(tmp_path: Path, relpath: str, text: str) -> Path:
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- 规则 1:函数内 import ---

def test_function_level_import(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def g():\n    import os\n    return os\n")
    assert any("function-level import" in v for v in scan_praxis_file(f))


# --- 规则 2:跨模块私有访问 ---

def test_cross_module_private_import(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("from praxis.store.log import _submit\n")
    assert any("private" in v for v in scan_praxis_file(f))


def test_cross_module_private_attribute_access(tmp_path):
    f = _write(tmp_path, "bad.py", "import praxis.store.log\n\npraxis.store.log._submit(1)\n")
    assert any("private" in v for v in scan_praxis_file(f))


def test_dunder_import_alias_not_flagged(tmp_path):
    # __init__/__version__ 等 dunder 名不算私有
    f = _write(tmp_path, "ok.py", "from praxis.store import __version__\n")
    assert scan_praxis_file(f) == []


# --- 规则 6:禁止 import 旧 runtime ---

def test_agent_runtime_import(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("import agent_runtime.tools.tool\n")
    assert any("agent_runtime" in v for v in scan_praxis_file(f))


def test_agent_runtime_from_import(tmp_path):
    f = _write(tmp_path, "bad.py", "from agent_runtime.harness import session\n")
    assert any("agent_runtime" in v for v in scan_praxis_file(f))


# --- 规则 3:UPDATE turns 只允许出现在 store/turns.py ---

def test_update_turns_sql_outside_store_turns(tmp_path):
    f = _write(tmp_path, "other.py", 'sql = "UPDATE turns SET status = ?"\n')
    assert any("UPDATE turns" in v for v in scan_praxis_file(f))


def test_update_turns_allowed_in_store_turns(tmp_path):
    f = _write(tmp_path, "store/turns.py", 'sql = "UPDATE turns SET status = ?"\n')
    assert scan_praxis_file(f) == []


# --- 规则 4:write_status() 调用只允许 core/statemachine.py ---

def test_write_status_outside_statemachine(tmp_path):
    f = _write(tmp_path, "other.py", "write_status(sid, status)\n")
    assert any("write_status" in v for v in scan_praxis_file(f))


def test_write_status_allowed_in_core_statemachine(tmp_path):
    f = _write(tmp_path, "core/statemachine.py", "write_status(sid, status)\n")
    assert scan_praxis_file(f) == []


# --- 规则 5:TurnStatus.INTERRUPTED 只能由 core/recovery.py 经 transition 写入 ---

def test_interrupted_transition_outside_recovery(tmp_path):
    f = _write(tmp_path, "other.py", "transition(sid, TurnStatus.INTERRUPTED)\n")
    assert any("INTERRUPTED" in v for v in scan_praxis_file(f))


def test_interrupted_transition_allowed_in_core_recovery(tmp_path):
    f = _write(tmp_path, "core/recovery.py", "transition(sid, TurnStatus.INTERRUPTED)\n")
    assert scan_praxis_file(f) == []


# --- 规则 7:Session(...) 构造只允许 composition.py ---

def test_session_construction_outside_composition(tmp_path):
    f = _write(tmp_path, "other.py", "sess = Session(store=store)\n")
    assert any("Session" in v for v in scan_praxis_file(f))


def test_session_allowed_in_composition(tmp_path):
    f = _write(tmp_path, "composition.py", "sess = Session(store=store)\n")
    assert scan_praxis_file(f) == []


# --- 干净文件不报任何违规 ---

def test_clean_file_has_no_violations(tmp_path):
    f = _write(tmp_path, "clean.py", "import os\n\n\ndef g():\n    return os.name\n")
    assert scan_praxis_file(f) == []
