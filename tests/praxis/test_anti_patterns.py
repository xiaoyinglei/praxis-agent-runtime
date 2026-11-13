"""阶段1 CI 硬规则:scripts/check_anti_patterns.py 的 praxis 扫描单元测试。

用 tmp_path 构造违规/合规 fixture,逐条验证 scan_praxis_file 的规则判定。
唯一写入点白名单按"相对路径后缀"匹配(如 store/turns.py),
因此 tmp fixture 也能模拟允许写入点,而不依赖真实 praxis 目录位置。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_anti_patterns as capi
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
    f = _write(tmp_path, "praxis/store/turns.py", 'sql = "UPDATE turns SET status = ?"\n')
    assert scan_praxis_file(f) == []


def test_update_turns_in_comment_not_flagged(tmp_path):
    # 规则只针对字符串字面量;注释里提到 UPDATE turns 不触发
    f = _write(tmp_path, "other.py", "# UPDATE turns 这种注释不应触发\nx = 1\n")
    assert scan_praxis_file(f) == []


# --- 规则 4:write_status() 调用只允许 core/statemachine.py ---

def test_write_status_outside_statemachine(tmp_path):
    f = _write(tmp_path, "other.py", "write_status(sid, status)\n")
    assert any("write_status" in v for v in scan_praxis_file(f))


def test_write_status_allowed_in_core_statemachine(tmp_path):
    f = _write(tmp_path, "praxis/core/statemachine.py", "write_status(sid, status)\n")
    assert scan_praxis_file(f) == []


# --- 规则 5:TurnStatus.INTERRUPTED 只能由 core/recovery.py 经 transition 写入 ---

def test_interrupted_transition_outside_recovery(tmp_path):
    f = _write(tmp_path, "other.py", "transition(sid, TurnStatus.INTERRUPTED)\n")
    assert any("INTERRUPTED" in v for v in scan_praxis_file(f))


def test_interrupted_transition_allowed_in_core_recovery(tmp_path):
    f = _write(tmp_path, "praxis/core/recovery.py", "transition(sid, TurnStatus.INTERRUPTED)\n")
    assert scan_praxis_file(f) == []


# --- 规则 7:Session(...) 构造只允许 composition.py ---

def test_session_construction_outside_composition(tmp_path):
    f = _write(tmp_path, "other.py", "sess = Session(store=store)\n")
    assert any("Session" in v for v in scan_praxis_file(f))


def test_session_allowed_in_composition(tmp_path):
    f = _write(tmp_path, "praxis/composition.py", "sess = Session(store=store)\n")
    assert scan_praxis_file(f) == []


def test_nested_composition_is_not_exempt(tmp_path):
    # 白名单严格等于"相对 praxis 根的路径":praxis/flow/composition.py 不豁免
    f = _write(tmp_path, "praxis/flow/composition.py", "sess = Session(store=store)\n")
    assert any("Session" in v for v in scan_praxis_file(f))


# --- 干净文件不报任何违规 ---

def test_clean_file_has_no_violations(tmp_path):
    f = _write(tmp_path, "clean.py", "import os\n\n\ndef g():\n    return os.name\n")
    assert scan_praxis_file(f) == []


# --- CLI:--praxis-only 只扫 praxis 树,跳过 rag 存量 ---

def test_praxis_only_flag_skips_rag(tmp_path, monkeypatch, capsys):
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    (rag_dir / "svc.py").write_text('URL = "https://api.deepseek.com/v1"\n', encoding="utf-8")
    praxis_dir = tmp_path / "praxis"
    praxis_dir.mkdir()
    (praxis_dir / "ok.py").write_text("import os\n\n\ndef g():\n    return os.name\n", encoding="utf-8")
    monkeypatch.setattr(capi, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(capi, "_RAG_DIR", rag_dir)
    monkeypatch.setattr(capi, "_PRAXIS_DIR", praxis_dir)

    assert capi.main([]) == 1  # 默认两棵树:rag 存量违规被计入
    assert "rag/svc.py" in capsys.readouterr().out

    assert capi.main(["--praxis-only"]) == 0  # 只扫 praxis:同一 rag 违规被跳过
    assert "praxis hard-rule scan clean" in capsys.readouterr().out


# ===========================================================================
# 阶段2 新增硬规则(Task 10):单一来源机械强制(设计 §8/findings S1/S2)
#
# 规则 8:沙箱 profile 构造字面量只允许 adapters/sandbox/(S1 唯一编译器);
# 规则 9:权限规则代数函数只允许 protocol/policy.py(唯一合并器);
# 规则 10:def decide() 只允许 core/permission.py(唯一决策入口);
# 规则 11:地板定义(floor_reason/FLOOR_ENTRIES)只允许 protocol/floor.py(唯一地板)。
# 误报纪律:规则 8 只钉代码字符串常量(AST),注释/文档串提及不算构造器
# ——真实树 protocol/sandbox.py docstring 解释 bwrap/sandbox-exec 必须不报。
# ===========================================================================

_PROFILE_MARKERS = ("(deny default)", "(version 1)", "sandbox-exec", "--unshare-all", "bwrap")


def test_second_profile_builder_rejected(tmp_path):
    # brief 形态:第二 profile 构造器的字面量出现在沙箱包之外
    f = _write(tmp_path, "praxis/core/evil.py", 'PROFILE = "(version 1)\\n(deny default)"\n')
    assert any("sandbox profile" in v for v in scan_praxis_file(f))


@pytest.mark.parametrize("marker", _PROFILE_MARKERS)
def test_profile_marker_literal_outside_sandbox_rejected(tmp_path, marker):
    f = _write(tmp_path, "praxis/tools/builtins/evil.py", f'FLAG = "{marker}"\n')
    vs = scan_praxis_file(f)
    assert any("S1 single source" in v and marker in v for v in vs)


def test_profile_marker_in_argv_list_rejected(tmp_path):
    # 后端 argv 拼接形态(sandbox-exec / --unshare-all 各报各的)
    f = _write(tmp_path, "praxis/core/evil.py", 'argv = ["/usr/bin/sandbox-exec", "--unshare-all"]\n')
    vs = scan_praxis_file(f)
    assert any("sandbox-exec" in v for v in vs)
    assert any("--unshare-all" in v for v in vs)


def test_sandbox_package_exempt(tmp_path):
    # adapters/sandbox/ 内的 profile 构造合法(唯一编译器持有面)
    f = _write(
        tmp_path,
        "praxis/adapters/sandbox/seatbelt.py",
        'PROFILE = "(version 1)\\n(deny default)"\nCMD = "/usr/bin/sandbox-exec"\n',
    )
    assert scan_praxis_file(f) == []


def test_sandbox_package_exempt_nested(tmp_path):
    # 豁免是整个包(含子目录),为阶段3 bwrap 后端预留同纪律
    f = _write(tmp_path, "praxis/adapters/sandbox/bwrap/backend.py", 'CMD = "bwrap"\n')
    assert scan_praxis_file(f) == []


def test_profile_marker_in_docstring_not_flagged(tmp_path):
    # 误报纪律:docstring 解释后端名词不是"第二构造器"
    f = _write(
        tmp_path,
        "praxis/protocol/sandbox.py",
        '"""macOS sandbox-exec / Linux bubblewrap(bwrap) backend."""\nBACKENDS = ("seatbelt",)\n',
    )
    assert scan_praxis_file(f) == []


def test_profile_marker_in_comment_not_flagged(tmp_path):
    f = _write(tmp_path, "praxis/composition.py", "# bwrap 延后,macOS 用 sandbox-exec\nx = 1\n")
    assert scan_praxis_file(f) == []


# --- 规则 9:规则代数只允许 protocol/policy.py ---

@pytest.mark.parametrize(
    "func",
    ("path_access", "resolve_policy", "tool_decision", "network_decision", "command_network_decision"),
)
def test_second_rule_algebra_rejected(tmp_path, func):
    f = _write(tmp_path, "praxis/core/merger.py", f"def {func}(policy, *rest):\n    return None\n")
    assert any("rule algebra outside protocol/policy.py" in v for v in scan_praxis_file(f))


def test_rule_algebra_allowed_in_policy(tmp_path):
    f = _write(
        tmp_path,
        "praxis/protocol/policy.py",
        "def path_access(policy, path):\n    return True\n\n\ndef resolve_policy(permissions):\n    return None\n",
    )
    assert scan_praxis_file(f) == []


def test_rule_algebra_rejected_in_protocol_sibling(tmp_path):
    # 白名单严格等于相对路径:protocol/ 内其他文件不豁免
    f = _write(tmp_path, "praxis/protocol/permission.py", "def tool_decision(policy, name):\n    return None\n")
    assert any("rule algebra" in v for v in scan_praxis_file(f))


def test_command_network_decision_rejected(tmp_path):
    # brief 第 5 个函数名也在列表内
    f = _write(tmp_path, "praxis/core/pipeline.py", "def command_network_decision(policy):\n    return None\n")
    assert any("single merger" in v for v in scan_praxis_file(f))


# --- 规则 10:decide() 只允许 core/permission.py ---

def test_second_decide_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/pipeline.py", "def decide(tool, use, ctx):\n    return None\n")
    assert any("permission decision outside core/permission.py" in v for v in scan_praxis_file(f))


def test_decide_method_rejected(tmp_path):
    # 类方法同名也是第二决策入口
    f = _write(tmp_path, "praxis/adapters/evil.py", "class Gate:\n    def decide(self, use):\n        return None\n")
    assert any("single entry" in v for v in scan_praxis_file(f))


def test_decide_allowed_in_core_permission(tmp_path):
    f = _write(tmp_path, "praxis/core/permission.py", "def decide(tool, use, ctx):\n    return None\n")
    assert scan_praxis_file(f) == []


# --- 规则 11:地板定义只允许 protocol/floor.py ---

def test_second_floor_reason_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/permission.py", "def floor_reason(path):\n    return None\n")
    assert any("floor definition outside protocol/floor.py" in v for v in scan_praxis_file(f))


def test_second_floor_entries_rejected(tmp_path):
    # 即使在豁免 profile 字面量的沙箱包内,地板清单也不得重定义(seatbelt 只能 import FLOOR_ENTRIES)
    f = _write(tmp_path, "praxis/adapters/sandbox/seatbelt.py", 'FLOOR_ENTRIES = (".git", ".praxis")\n')
    assert any("single floor" in v for v in scan_praxis_file(f))


def test_annotated_floor_entries_rejected(tmp_path):
    # 真实形态是带注解赋值:FLOOR_ENTRIES: tuple[...] = (...)
    f = _write(tmp_path, "praxis/protocol/models.py", 'FLOOR_ENTRIES: tuple[str, ...] = (".git",)\n')
    assert any("floor definition" in v for v in scan_praxis_file(f))


def test_bare_annotation_not_a_definition(tmp_path):
    # 无 value 的裸注解不是地板清单定义
    f = _write(tmp_path, "praxis/protocol/models.py", "FLOOR_ENTRIES: tuple[str, ...]\n")
    assert scan_praxis_file(f) == []


def test_floor_defs_allowed_in_floor_module(tmp_path):
    f = _write(
        tmp_path,
        "praxis/protocol/floor.py",
        'FLOOR_ENTRIES: tuple[str, ...] = (".git",)\n\n\ndef floor_reason(path):\n    return None\n',
    )
    assert scan_praxis_file(f) == []


# --- 真实树:新规则零误报(与 CI 门同源,钉住 adapters/sandbox 豁免+注释/docstring 排除) ---

def test_real_praxis_tree_clean_under_new_rules():
    assert capi.scan_praxis_tree(capi._PRAXIS_DIR) == []
