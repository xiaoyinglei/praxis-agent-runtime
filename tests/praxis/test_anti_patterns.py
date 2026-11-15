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


# ===========================================================================
# 阶段3A 新增硬规则(Task 17):单一来源继续收口
#
# ⑨ provider SDK 面只在 adapters/models/;⑩ token 测量只在 tokenization.py;
# ⑪ 重试机制/策略分层两侧各自唯一;⑫ 全局 SQL 只在两个 store;
# ⑬ 凭据存储只在 credentials.py;⑭ artifact put 只在 artifacts.py;
# ⑮ MAX_PLAN_STEPS 定义只在 protocol/plan.py。
# 机制沿用既有纪律:严格相对路径白名单 + AST(代码字符串排除 docstring)。
# ===========================================================================


# --- 规则⑨:provider SDK 面只在 adapters/models/ ---

def test_chat_completions_create_outside_models_rejected(tmp_path):
    f = _write(
        tmp_path,
        "praxis/tools/evil.py",
        "async def ask(client):\n    return await client.chat.completions.create(model='m')\n",
    )
    assert any("adapters/models" in v and "chat.completions.create" in v for v in scan_praxis_file(f))


def test_async_openai_construction_outside_models_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", "CLIENT = AsyncOpenAI(api_key='k')\n")
    assert any("AsyncOpenAI" in v and "adapters/models" in v for v in scan_praxis_file(f))


def test_event_stream_literal_outside_models_rejected(tmp_path):
    f = _write(tmp_path, "praxis/composition.py", 'HEADERS = {"Accept": "text/event-stream"}\n')
    assert any("text/event-stream" in v for v in scan_praxis_file(f))


def test_model_surface_allowed_in_adapters_models(tmp_path):
    f = _write(
        tmp_path,
        "praxis/adapters/models/ok.py",
        'HEADERS = {"Accept": "text/event-stream"}\n'
        "CLIENT = AsyncOpenAI(api_key='k')\n"
        "async def ask(client):\n    return await client.chat.completions.create(model='m')\n",
    )
    assert scan_praxis_file(f) == []


def test_event_stream_in_docstring_not_flagged(tmp_path):
    f = _write(
        tmp_path,
        "praxis/protocol/sandbox.py",
        '"""Adapters may request text/event-stream from providers."""\nX = 1\n',
    )
    assert scan_praxis_file(f) == []


# --- 规则⑩:token 测量只在 adapters/models/tokenization.py ---

@pytest.mark.parametrize("func", ("measure_request", "count_text_tokens"))
def test_token_measurement_outside_tokenization_rejected(tmp_path, func):
    f = _write(tmp_path, "praxis/core/evil.py", f"def {func}(x):\n    return 0\n")
    assert any("tokenization.py" in v for v in scan_praxis_file(f))


def test_token_measurement_allowed_in_tokenization(tmp_path):
    f = _write(
        tmp_path,
        "praxis/adapters/models/tokenization.py",
        "def count_text_tokens(text):\n    return 0\n\n\ndef measure_request(req):\n    return 0\n",
    )
    assert scan_praxis_file(f) == []


# --- 规则⑪:重试机制(adapters)与策略(protocol)各自唯一 ---

@pytest.mark.parametrize("func", ("classify_failure", "stream_with_retry", "make_retry_stream"))
def test_retry_mechanism_outside_adapters_retry_rejected(tmp_path, func):
    f = _write(tmp_path, "praxis/core/evil.py", f"def {func}(*args):\n    return None\n")
    assert any("adapters/models/retry.py" in v for v in scan_praxis_file(f))


def test_retry_mechanism_allowed_in_adapters_retry(tmp_path):
    f = _write(
        tmp_path,
        "praxis/adapters/models/retry.py",
        "def classify_failure(exc):\n    return None\n\n\n"
        "def stream_with_retry(factory):\n    return None\n\n\n"
        "def make_retry_stream(policy):\n    return None\n",
    )
    assert scan_praxis_file(f) == []


@pytest.mark.parametrize(
    "definition",
    (
        "class RetryPolicy:\n    pass\n",
        "class ModelFailure:\n    pass\n",
        "class RetryPlan:\n    pass\n",
        "class NoRetryPlan:\n    pass\n",
        "class AttemptRecord:\n    pass\n",
        "class ModelUnavailableError(RuntimeError):\n    pass\n",
        "def clock_floor(records, now):\n    return now\n",
    ),
)
def test_retry_policy_definition_outside_protocol_retry_rejected(tmp_path, definition):
    f = _write(tmp_path, "praxis/core/evil.py", definition)
    assert any("protocol/retry.py" in v for v in scan_praxis_file(f))


def test_retry_policy_definition_allowed_in_protocol_retry(tmp_path):
    f = _write(
        tmp_path,
        "praxis/protocol/retry.py",
        "class RetryPolicy:\n    pass\n\n\n"
        "class ModelFailure:\n    pass\n\n\n"
        "class RetryPlan:\n    pass\n\n\n"
        "class NoRetryPlan:\n    pass\n\n\n"
        "class AttemptRecord:\n    pass\n\n\n"
        "class ModelUnavailableError(RuntimeError):\n    pass\n\n\n"
        "def clock_floor(records, now):\n    return now\n",
    )
    assert scan_praxis_file(f) == []


def test_retry_policy_decide_method_still_rejected(tmp_path):
    # 规则③(def decide 只在 core/permission.py)保持零豁免:策略对象不得开路径豁免
    f = _write(
        tmp_path,
        "praxis/protocol/retry.py",
        "class RetryPolicy:\n    def decide(self, failure):\n        return None\n",
    )
    assert any("core/permission.py" in v for v in scan_praxis_file(f))


# --- 规则⑫:全局 SQL 只在 store/log.py 与 adapters/config/store.py ---

def test_create_table_outside_sql_files_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", 'DDL = "CREATE TABLE IF NOT EXISTS x(id TEXT)"\n')
    assert any("store/log.py" in v and "CREATE TABLE" in v for v in scan_praxis_file(f))


def test_sqlite_connect_outside_sql_files_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", "conn = sqlite3.connect('x.db')\n")
    assert any("sqlite3.connect" in v for v in scan_praxis_file(f))


@pytest.mark.parametrize("relpath", ("praxis/store/log.py", "praxis/adapters/config/store.py"))
def test_global_sql_allowed_in_the_two_stores(tmp_path, relpath):
    f = _write(
        tmp_path,
        relpath,
        'DDL = "CREATE TABLE IF NOT EXISTS x(id TEXT)"\nconn = sqlite3.connect("x.db")\n',
    )
    assert scan_praxis_file(f) == []


def test_create_table_in_docstring_not_flagged(tmp_path):
    f = _write(tmp_path, "praxis/store/artifacts.py", '"""No CREATE TABLE here: artifacts are files."""\nX = 1\n')
    assert scan_praxis_file(f) == []


# --- 规则⑬:凭据存储只在 adapters/config/credentials.py ---

def test_find_generic_password_outside_credentials_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", 'CMD = ["/usr/bin/security", "find-generic-password"]\n')
    assert any("credentials.py" in v for v in scan_praxis_file(f))


def test_security_subprocess_outside_credentials_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", 'CMD = ["security", "-i"]\n')
    assert any("credentials.py" in v for v in scan_praxis_file(f))


def test_credential_file_mode_outside_credentials_rejected(tmp_path):
    f = _write(
        tmp_path,
        "praxis/core/evil.py",
        'fd = os.open("credentials/models.json", os.O_RDONLY | os.O_CREAT, 0o600)\n',
    )
    assert any("credentials.py" in v and "0o600" in v for v in scan_praxis_file(f))


def test_credential_storage_allowed_in_credentials_module(tmp_path):
    f = _write(
        tmp_path,
        "praxis/adapters/config/credentials.py",
        "import os\n"
        "CMD = ['/usr/bin/security', 'find-generic-password']\n"
        "fd = os.open('credentials/models.json', os.O_RDONLY, 0o600)\n",
    )
    assert scan_praxis_file(f) == []


def test_private_mode_without_credential_context_not_flagged(tmp_path):
    # 0o600 本身不是凭据信号:artifact/后台 marker 也用它
    f = _write(tmp_path, "praxis/store/artifacts.py", "os.chmod(tmp, 0o600)\n")
    assert scan_praxis_file(f) == []


# --- 规则⑭:artifact put() 只在 store/artifacts.py ---

def test_put_outside_artifacts_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", "def put(data):\n    return data\n")
    assert any("store/artifacts.py" in v for v in scan_praxis_file(f))


def test_put_allowed_in_artifacts(tmp_path):
    f = _write(tmp_path, "praxis/store/artifacts.py", "class S:\n    def put(self, data):\n        return data\n")
    assert scan_praxis_file(f) == []


# --- 规则⑮:MAX_PLAN_STEPS 定义只在 protocol/plan.py ---

def test_max_plan_steps_outside_plan_rejected(tmp_path):
    f = _write(tmp_path, "praxis/tools/builtins/evil.py", "MAX_PLAN_STEPS = 20\n")
    assert any("protocol/plan.py" in v for v in scan_praxis_file(f))


def test_annotated_max_plan_steps_outside_plan_rejected(tmp_path):
    # 沿用规则 8 的 AnnAssign 覆盖面
    f = _write(tmp_path, "praxis/core/evil.py", "MAX_PLAN_STEPS: int = 20\n")
    assert any("protocol/plan.py" in v for v in scan_praxis_file(f))


def test_bare_max_plan_steps_annotation_not_flagged(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", "MAX_PLAN_STEPS: int\n")
    assert scan_praxis_file(f) == []


def test_max_plan_steps_allowed_in_plan_module(tmp_path):
    f = _write(tmp_path, "praxis/protocol/plan.py", "MAX_PLAN_STEPS = 20\n")
    assert scan_praxis_file(f) == []


# --- 真实树:新规则零误报(与 CI 门同源,钉住 adapters/sandbox 豁免+注释/docstring 排除) ---

# ===========================================================================
# 阶段3B 新增硬规则(Task 20):CI 单一来源收口
#
# ⑯ mcp__ 字面量只允许 adapters/mcp/names.py;
# ⑰ S7 argv env 重置形态(/usr/bin/env、env -i)全仓禁止;
# ⑱ B33 宿主 env 直接进子进程 env 全仓禁止;
# ⑲ child_env/signal_group/... 唯一定义点;
# ⑳ 宿主配置文件读取面(host_config_dir / PRAXIS_* env 名)白名单。
# ===========================================================================


# --- 规则⑯:MCP 工具名前缀只允许 names.py 唯一拼装 ---

def test_mcp_name_literal_outside_names_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", 'PREFIX = "mcp__probe__echo"\n')
    assert any("MCP tool-name literal" in v for v in scan_praxis_file(f))


def test_mcp_name_literal_allowed_in_names_module(tmp_path):
    f = _write(
        tmp_path,
        "praxis/adapters/mcp/names.py",
        'def canonical_tool_name(server, tool):\n'
        '    return f"mcp__{server}__{tool}"\n\n\n'
        'def global_wildcard():\n    return "mcp__*"\n',
    )
    assert scan_praxis_file(f) == []


def test_mcp_name_literal_in_docstring_not_flagged(tmp_path):
    f = _write(
        tmp_path,
        "praxis/adapters/mcp/output.py",
        '"""canonical 名为 ``mcp__server__tool``;``mcp__*`` 是全域通配。"""\nX = 1\n',
    )
    assert scan_praxis_file(f) == []


# --- 规则⑰:S7 argv env 重置形态全仓禁止 ---

def test_s7_env_reset_shape_rejected(tmp_path):
    f = _write(
        tmp_path,
        "praxis/adapters/mcp/lifecycle.py",
        'argv = ["/usr/bin/env", "-i", "SECRET_MCP_KEY=value", "server"]\n',
    )
    assert any("S7 argv env reset" in v for v in scan_praxis_file(f))


def test_s7_env_binary_literal_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", 'RESET = "/usr/bin/env"\n')
    assert any("/usr/bin/env" in v and "S7" in v for v in scan_praxis_file(f))


def test_s7_env_name_reset_shape_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", 'argv = ["env", "-i", "K=V", "cmd"]\n')
    assert any("S7 argv env reset" in v for v in scan_praxis_file(f))


def test_s7_mentions_in_docstring_not_flagged(tmp_path):
    # 真实树 lifecycle.py 用 docstring 解释 S7 病灶:文档不是构造器
    f = _write(
        tmp_path,
        "praxis/adapters/mcp/lifecycle.py",
        '"""旧 ``/usr/bin/env -i KEY=VALUE ... cmd`` argv 重置形态已删除。"""\nX = 1\n',
    )
    assert scan_praxis_file(f) == []


def test_security_stdin_flag_is_not_s7_reset_shape(tmp_path):
    # credentials 的 ``security -i`` 是 stdin 模式,没有 env 可执行名共现
    f = _write(
        tmp_path,
        "praxis/adapters/config/credentials.py",
        'CMD = ["/usr/bin/security", "-i"]\n',
    )
    assert scan_praxis_file(f) == []


# --- 规则⑱:B33 宿主 env 直接进子进程 ---

@pytest.mark.parametrize(
    "source",
    (
        'proc = create_subprocess_exec("x", env={**os.environ})\n',
        'env = os.environ.copy()\nproc = create_subprocess_exec("x", env=env)\n',
        'env = dict(os.environ)\nproc = create_subprocess_exec("x", env=env)\n',
        'proc = create_subprocess_exec("x", env=os.environ)\n',
        # BitOr 合并(现代惯用写法,同样是把宿主 env 带进子进程)
        'proc = create_subprocess_exec("x", env=os.environ | {"PATH": "/bin"})\n',
        'proc = create_subprocess_exec("x", env={"PATH": "/bin"} | os.environ)\n',
        # dict-comp over items()
        'env = {key: value for key, value in os.environ.items()}\n'
        'proc = create_subprocess_exec("x", env=env)\n',
        'proc = create_subprocess_exec("x", env=dict(os.environ.items()))\n',
        # 一层别名再展开/再 copy/dict
        'source = os.environ\nproc = create_subprocess_exec("x", env={**source})\n',
        'source = os.environ.copy()\nproc = create_subprocess_exec("x", env=dict(source))\n',
    ),
)
def test_host_environment_inheritance_rejected(tmp_path, source):
    f = _write(tmp_path, "praxis/tools/builtins/evil.py", source)
    assert any("B33" in v for v in scan_praxis_file(f))


def test_plain_dict_union_is_not_environment_inheritance(tmp_path):
    # 两个常量 dict 的 | 合并没有宿主 env 源,不得误报
    f = _write(
        tmp_path,
        "praxis/tools/builtins/evil.py",
        'proc = create_subprocess_exec("x", env={"PATH": "/bin"} | {"LANG": "C"})\n',
    )
    assert scan_praxis_file(f) == []


def test_environment_alias_conservative_after_reassignment(tmp_path):
    """记录保守近似:单遍无数据流,别名后重绑定为常量仍命中(B33 失败方向)。"""
    f = _write(
        tmp_path,
        "praxis/tools/builtins/evil.py",
        'source = os.environ\nsource = {"PATH": "/bin"}\n'
        'proc = create_subprocess_exec("x", env={**source})\n',
    )
    assert any("B33" in v for v in scan_praxis_file(f))


def test_constant_environment_whitelist_allowed(tmp_path):
    f = _write(
        tmp_path,
        "praxis/adapters/hooks/runner.py",
        'import os\nproc = create_subprocess_exec("x", env={"PATH": os.defpath})\n',
    )
    assert scan_praxis_file(f) == []


def test_b33_pattern_in_docstring_not_flagged(tmp_path):
    f = _write(
        tmp_path,
        "praxis/tools/builtins/child_env.py",
        '"""Never pass ``env={**os.environ}`` to a child process."""\nX = 1\n',
    )
    assert scan_praxis_file(f) == []


# --- 规则⑲:单一定义点 ---

@pytest.mark.parametrize(
    "func",
    (
        "child_env",
        "signal_group",
        "await_main_exit",
        "drain_stream",
        "skill_fingerprint",
        "atomic_write_bytes",
        "atomic_write_text",
        "load_host_hooks",
        "spawn_hook",
        "connect_mcp_servers",
        "web_configuration_path",
    ),
)
def test_single_definition_outside_its_site_rejected(tmp_path, func):
    f = _write(tmp_path, "praxis/core/evil.py", f"def {func}(*args):\n    return None\n")
    assert any(f"def {func}()" in v and "single definition" in v for v in scan_praxis_file(f))


def test_single_definitions_allowed_at_their_sites(tmp_path):
    f = _write(
        tmp_path,
        "praxis/tools/builtins/process.py",
        "def signal_group(pgid, sig):\n    return True\n\n\n"
        "async def await_main_exit(proc, timeout):\n    return 0\n\n\n"
        "async def drain_stream(proc):\n    return None\n",
    )
    assert scan_praxis_file(f) == []
    paths = _write(
        tmp_path,
        "praxis/tools/builtins/paths.py",
        "def atomic_write_bytes(path, data, *, mode=None):\n    return None\n\n\n"
        "def atomic_write_text(path, content):\n    return None\n",
    )
    assert scan_praxis_file(paths) == []
    hooks = _write(
        tmp_path,
        "praxis/adapters/hooks/config.py",
        "def load_host_hooks(workspace, path=None):\n    return ()\n",
    )
    assert scan_praxis_file(hooks) == []
    runner = _write(
        tmp_path,
        "praxis/adapters/hooks/runner.py",
        "async def spawn_hook(spec, request, *, kill_group=None):\n    return None\n",
    )
    assert scan_praxis_file(runner) == []
    skills = _write(
        tmp_path,
        "praxis/tools/builtins/skills.py",
        "def skill_fingerprint(frontmatter, body):\n    return ''\n",
    )
    assert scan_praxis_file(skills) == []
    child = _write(
        tmp_path,
        "praxis/tools/builtins/child_env.py",
        "def child_env(scratch_dir, workspace):\n    return {}\n",
    )
    assert scan_praxis_file(child) == []
    limits = _write(
        tmp_path,
        "praxis/adapters/web/limits.py",
        "def web_configuration_path():\n    return None\n",
    )
    assert scan_praxis_file(limits) == []


def test_connect_mcp_servers_facade_allowed_in_composition(tmp_path):
    # manager.py 是唯一实现;composition.py 是唯一的注入绑定 facade。
    manager = _write(
        tmp_path,
        "praxis/adapters/mcp/manager.py",
        "async def connect_mcp_servers(workspace, **kwargs):\n    return None\n",
    )
    assert scan_praxis_file(manager) == []
    facade = _write(
        tmp_path,
        "praxis/composition.py",
        "async def connect_mcp_servers(workspace, **kwargs):\n    return None\n",
    )
    assert scan_praxis_file(facade) == []


# --- 规则⑳:宿主配置文件读取面白名单 ---

def test_host_config_dir_outside_config_adapters_rejected(tmp_path):
    f = _write(tmp_path, "praxis/core/evil.py", 'path = host_config_dir() / "web.json"\n')
    assert any("host configuration path" in v for v in scan_praxis_file(f))


def test_host_config_env_literal_outside_config_adapters_rejected(tmp_path):
    f = _write(tmp_path, "praxis/interfaces/cli/evil.py", 'ENV = "PRAXIS_MCP_CONFIG"\n')
    assert any("PRAXIS_MCP_CONFIG" in v and "host configuration env" in v for v in scan_praxis_file(f))


@pytest.mark.parametrize(
    "relpath",
    (
        "praxis/adapters/config/store.py",
        "praxis/adapters/config/credentials.py",
        "praxis/adapters/hooks/config.py",
        "praxis/adapters/web/limits.py",
        "praxis/adapters/web/search.py",
        "praxis/adapters/mcp/config.py",
        "praxis/protocol/floor.py",
    ),
)
def test_host_config_reader_allowed_at_config_layers(tmp_path, relpath):
    f = _write(
        tmp_path,
        relpath,
        'ENV = "PRAXIS_WEB_SEARCH_PROVIDER"\npath = host_config_dir() / "config.json"\n',
    )
    assert scan_praxis_file(f) == []


def test_real_praxis_tree_clean_under_3b_rules():
    assert capi.scan_praxis_tree(capi._PRAXIS_DIR) == []
