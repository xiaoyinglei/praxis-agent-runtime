"""M2:``praxis agent list/show`` 子身份 discoverability(Q3 直面 resume 前提)。

子注册表 = child_allocation items 折叠(无表列)。本测试手工落 allocation +
子根 turn 事实(与 core.agents 折叠同口径),经真 CliRunner 验证 list/show
从事实重建子的 thread/turn/state/workspace_mode/workspace,并给出 resume 指引。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from praxis.core.events import EventBus
from praxis.core.statemachine import TurnStatus, transition
from praxis.interfaces.cli import app
from praxis.protocol.budget import UNLIMITED, allocation_payload
from praxis.protocol.items import ItemKind
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn

runner = CliRunner()
_CLEAN_ENV = {
    "PRAXIS_MODEL_ID": None,
    "PRAXIS_MODEL": None,
    "PRAXIS_BASE_URL": None,
    "PRAXIS_API_KEY": None,
    "OPENAI_API_KEY": None,
}


def _invoke(*args: str) -> Any:
    return runner.invoke(app, list(args), env=_CLEAN_ENV)


def _seed_child(workspace: Path, *, state: str = "completed") -> tuple[str, str]:
    db = workspace / ".praxis" / "praxis.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    store = Store(db)
    try:
        store.ensure_thread("parent")
        store.ensure_thread("child-t")
        create_turn(store, turn_id="parent-turn", thread_id="parent", binding={})
        create_turn(store, turn_id="child-turn", thread_id="child-t", binding={})
        if state == "completed":
            transition(store, EventBus(), "child-turn", TurnStatus.COMPLETED, reason="test")
        store.append_items(
            "parent",
            "parent-turn",
            [
                ItemEntry(
                    kind=ItemKind.CHILD_ALLOCATION,
                    payload=allocation_payload(
                        allocation_id="alloc-1",
                        parent_turn_id="parent-turn",
                        child_thread_id="child-t",
                        child_root_turn_id="child-turn",
                        workspace_mode="read_only",
                        workspace=str(workspace),
                        grant=UNLIMITED,
                        requested=None,
                        source="model",
                    ),
                )
            ],
        )
    finally:
        store.close()
    return "child-t", "child-turn"


def test_agent_list_folds_child_identity(workspace: Path) -> None:
    child_thread, child_turn = _seed_child(workspace)
    result = _invoke("agent", "list", "--workspace", str(workspace), "--thread", "parent", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert len(data) == 1
    assert data[0]["child_thread_id"] == child_thread
    assert data[0]["child_root_turn_id"] == child_turn
    assert data[0]["state"] == "completed"
    assert data[0]["workspace_mode"] == "read_only"
    assert data[0]["workspace"] == str(workspace)

    human = _invoke("agent", "list", "--workspace", str(workspace), "--thread", "parent")
    assert human.exit_code == 0, human.output
    assert child_thread in human.output
    assert "read_only" in human.output


def test_agent_list_empty_thread(workspace: Path) -> None:
    result = _invoke("agent", "list", "--workspace", str(workspace), "--thread", "parent")
    assert result.exit_code == 0, result.output
    assert "no subagent children" in result.output


def test_agent_show_detail_and_resume_hint(workspace: Path) -> None:
    child_thread, child_turn = _seed_child(workspace)
    result = _invoke(
        "agent", "show", child_thread, "--workspace", str(workspace), "--thread", "parent", "--json"
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["child_thread_id"] == child_thread
    assert data["state"] == "completed"
    assert data["child_root_turn_id"] == child_turn

    human = _invoke("agent", "show", child_thread, "--workspace", str(workspace), "--thread", "parent")
    assert human.exit_code == 0, human.output
    assert f"praxis resume {child_turn} --thread {child_thread}" in human.output


def test_agent_show_unknown_child_is_usage_error(workspace: Path) -> None:
    _seed_child(workspace)
    result = _invoke("agent", "show", "ghost", "--workspace", str(workspace), "--thread", "parent")
    assert result.exit_code == 3
    assert "unknown child thread" in result.output


def test_agent_help_mounted_on_root() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0, result.output
    assert "agent" in result.output
    assert "memory" in result.output  # M5:root help 含 memory 子组