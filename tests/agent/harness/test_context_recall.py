from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

from agent_runtime.harness import RolloutStore, ToolOrchestrator
from agent_runtime.harness.context_recall import create_context_recall_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin


def test_tool_recall_returns_content_without_recursive_payload_encoding(tmp_path):
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
    from tests.agent.harness.test_compaction_consistency import start

    with RolloutStore(tmp_path / "content.db") as store:
        _, turn = start(store, tmp_path)
        content = 'RULE: preserve tenant identity.\n{"sample":123456789012345}'
        item = seed(store, turn_id=turn.turn_id, kind="tool_result", payload={
            "tool_call_id": "read", "tool_name": "read_file", "is_error": False,
            "structured_content": {"content": content, "path": "spec.md"},
            "model_content": json.dumps({"structured_content": {"content": content}}),
        })
        result = _read(store, tmp_path, turn.turn_id, item_id=item.item_id)
        page = json.loads(result.content[0].data["text"])
        assert page["text"] == content
        assert page["format"] == "tool_content"
        raw = _read(store, tmp_path, turn.turn_id, item_id=item.item_id, view="raw", max_chars=4000)
        assert json.loads(json.loads(raw.content[0].data["text"])["text"]) == dict(item.payload)


def _read(store, workspace, turn_id, **arguments):
    tool = create_context_recall_tool(store)
    orchestrator = ToolOrchestrator(
        store=store,
        tools={"read_context": tool},
        execution_context=ToolExecutionContext(workspace_root=workspace, cwd=workspace),
    )
    return asyncio.run(
        orchestrator.execute(
            turn_id=turn_id,
            call=ToolCall(
                tool_call_id=f"call-{uuid4().hex}",
                tool_name="read_context",
                arguments=arguments,
                origin=ToolCallOrigin(
                    request_id="request", toolset_revision="revision", exposed_tool_names=("read_context",)
                ),
            ),
        )
    )


def test_recall_pages_original_payload_through_orchestrator(tmp_path: Path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="原始🙂\n" * 2000, binding_manifest={})
        item = store.list_items(turn.turn_id)[0]
        offset, chunks = 0, []
        while True:
            result = _read(store, tmp_path, turn.turn_id, item_id=item.item_id, offset=offset, max_chars=4000)
            assert not result.is_error and not result.truncated
            page = json.loads(result.content[0].data["text"])
            chunks.append(page["text"])
            assert len(page["text"]) <= 4000
            if not page["truncated"]:
                assert page["next_offset"] is None
                break
            assert page["next_offset"] == offset + len(page["text"])
            offset = page["next_offset"]
        assert json.loads("".join(chunks)) == dict(item.payload)
        assert all(op.effects == () and op.status == "succeeded" for op in store.list_tool_operations(turn.turn_id))
        assert store.verify().valid


def test_recall_visibility_respects_fork_and_other_threads(tmp_path: Path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = store.create_thread(workspace=tmp_path)
        first = store.start_turn(thread_id=parent.thread_id, user_message="visible", binding_manifest={})
        visible = store.list_items(first.turn_id)[0]
        store.complete_turn(turn_id=first.turn_id, answer="done")
        child = store.fork_thread(from_turn_id=first.turn_id)
        later = store.start_turn(thread_id=parent.thread_id, user_message="parent future secret", binding_manifest={})
        hidden = store.list_items(later.turn_id)[0]
        current = store.start_turn(thread_id=child.thread_id, user_message="recall", binding_manifest={})
        other = store.create_thread(workspace=tmp_path)
        other_turn = store.start_turn(thread_id=other.thread_id, user_message="other secret", binding_manifest={})
        other_item = store.list_items(other_turn.turn_id)[0]
        assert not _read(store, tmp_path, current.turn_id, item_id=visible.item_id).is_error
        for item_id in (hidden.item_id, other_item.item_id, "missing", "/etc/passwd"):
            result = _read(store, tmp_path, current.turn_id, item_id=item_id)
            assert result.is_error and result.error_code == "context_item_unavailable"
            assert "secret" not in str(result)


def test_recall_requires_runtime_turn_identity(tmp_path: Path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        tool = create_context_recall_tool(store)
        with pytest.raises(RuntimeError, match="Harness Turn"):
            tool.run(tool.validate_input({"item_id": "anything"}))


def test_recall_exposes_chronology_across_turns(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        first = store.start_turn(thread_id=thread.thread_id, user_message="TTL=17", binding_manifest={})
        older = store.list_items(first.turn_id)[0]
        store.complete_turn(turn_id=first.turn_id, answer="decision recorded")
        second = store.start_turn(thread_id=thread.thread_id, user_message="TTL=23 supersedes 17", binding_manifest={})
        newer = store.list_items(second.turn_id)[0]
        pages = [
            json.loads(_read(store, tmp_path, second.turn_id, item_id=item.item_id).content[0].data["text"])
            for item in (older, newer)
        ]
        assert pages[0]["history_index"] < pages[1]["history_index"]


@pytest.mark.parametrize(
    "arguments", [{"offset": -1}, {"max_chars": 0}, {"max_chars": 4001}, {"offset": True}, {"path": "file"}]
)
def test_recall_rejects_invalid_pagination(tmp_path: Path, arguments):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        tool = create_context_recall_tool(store)
        with pytest.raises(ValueError):
            tool.validate_input({"item_id": "item", **arguments})


def test_recall_end_offset_and_byte_budget(tmp_path: Path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message='\x00🙂"\\' * 4000, binding_manifest={})
        item = store.list_items(turn.turn_id)[0]
        result = _read(store, tmp_path, turn.turn_id, item_id=item.item_id, max_chars=4000)
        assert not result.truncated
        page = json.loads(result.content[0].data["text"])
        assert page["next_offset"] == 4000
        end = _read(store, tmp_path, turn.turn_id, item_id=item.item_id, offset=page["total_chars"])
        assert not end.is_error
        end_page = json.loads(end.content[0].data["text"])
        assert end_page["text"] == "" and end_page["next_offset"] is None and not end_page["truncated"]
        invalid = _read(store, tmp_path, turn.turn_id, item_id=item.item_id, offset=page["total_chars"] + 1)
        assert invalid.error_code == "context_offset_out_of_range"


def test_archive_catalog_can_recover_original_without_summary_retaining_ids(tmp_path: Path):
    from agent_runtime.harness import RolloutContextManager
    from tests.agent.harness.test_compaction_consistency import long_history, start

    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        manager.commit_compaction(manager.semantic_candidate(turn.turn_id, source_hash=source_hash, summary="Done."))
        result = _read(store, tmp_path, turn.turn_id, item_id="history", max_chars=4000)
        page = json.loads(result.content[0].data["text"])
        catalog = json.loads(page["text"])
        reference = next(item["item_id"] for item in catalog if item["kind"] == "agent_message")
        original = _read(store, tmp_path, turn.turn_id, item_id=reference, max_chars=4000)
        payload = json.loads(json.loads(original.content[0].data["text"])["text"])
        assert payload["text"].startswith("finding 0:")


def test_history_query_searches_archived_content_without_knowing_item_id(tmp_path: Path):
    from agent_runtime.harness import RolloutContextManager
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
    from tests.agent.harness.test_compaction_consistency import start

    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        seed(
            store,
            turn_id=turn.turn_id,
            kind="agent_message",
            payload={"text": "noise " * 1000 + "Helios HX-7294-KAPPA" + " noise" * 1000},
        )
        source_item = store.list_items(turn.turn_id)[-1]
        manager = RolloutContextManager(store)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        manager.commit_compaction(
            manager.semantic_candidate(
                turn.turn_id,
                source_hash=source_hash,
                summary="The earlier reference code is in the archive.",
            )
        )
        result = _read(store, tmp_path, turn.turn_id, item_id="history", query="helios", max_chars=200)
        page = json.loads(result.content[0].data["text"])
        assert page["matched"] is True
        assert page["matched_item_id"] == source_item.item_id
        assert "HX-7294-KAPPA" in page["text"]
        assert page["search_scope"] == "visible_archived_originals"
        absent = _read(store, tmp_path, turn.turn_id, item_id="history", query="not-present", max_chars=200)
        missing = json.loads(absent.content[0].data["text"])
        assert missing["matched"] is False and missing["next_index"] is None
        assert missing["searched_to_index"] >= page["history_index"]


def test_history_search_respects_fork_visibility(tmp_path: Path):
    from agent_runtime.harness import RolloutContextManager

    with RolloutStore(tmp_path / "r.db") as store:
        first_thread = store.create_thread(workspace=tmp_path)
        first = store.start_turn(thread_id=first_thread.thread_id, user_message="parent public", binding_manifest={})
        store.complete_turn(turn_id=first.turn_id, answer="visible result " * 1000)
        fork = store.fork_thread(from_turn_id=first.turn_id)
        later = store.start_turn(thread_id=first_thread.thread_id, user_message="secret-sentinel", binding_manifest={})
        store.complete_turn(turn_id=later.turn_id, answer="secret-sentinel")
        child = store.start_turn(thread_id=fork.thread_id, user_message="continue", binding_manifest={})
        manager = RolloutContextManager(store)
        source_hash, _ = manager.semantic_source(child.turn_id)
        manager.commit_compaction(
            manager.semantic_candidate(
                child.turn_id,
                source_hash=source_hash,
                summary="Prior public work complete.",
            )
        )
        result = _read(store, tmp_path, child.turn_id, item_id="history", query="secret-sentinel")
        page = json.loads(result.content[0].data["text"])
        assert page["matched"] is False
        assert "secret-sentinel" not in page["text"]


def test_query_finds_detail_between_skipped_pages_and_proves_absence(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="noise " * 1000 + "Helios code HX-7294-KAPPA" + " noise" * 1000,
            binding_manifest={},
        )
        item = store.list_items(turn.turn_id)[0]
        found = _read(store, tmp_path, turn.turn_id, item_id=item.item_id, query="helios", max_chars=200)
        page = json.loads(found.content[0].data["text"])
        assert page["matched"] is True and "HX-7294-KAPPA" in page["text"]
        assert page["match_offset"] > 4000
        absent = _read(store, tmp_path, turn.turn_id, item_id=item.item_id, query="nonexistent")
        missing = json.loads(absent.content[0].data["text"])
        assert missing["matched"] is False
        assert missing["searched_from"] == 0 and missing["searched_to"] == missing["total_chars"]


def test_archive_search_cursor_crosses_bounded_scan_without_false_absence(tmp_path: Path):
    from agent_runtime.harness import RolloutContextManager
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
    from tests.agent.harness.test_compaction_consistency import start

    with RolloutStore(tmp_path / "pages.db") as store:
        _, turn = start(store, tmp_path)
        for index in range(70):
            seed(
                store,
                turn_id=turn.turn_id,
                kind="agent_message",
                payload={"text": f"record {index}: " + ("needle-at-end" if index == 69 else "no match") * 20},
            )
        manager = RolloutContextManager(store)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        manager.commit_compaction(manager.semantic_candidate(turn.turn_id, source_hash=source_hash, summary="Archive."))
        first = json.loads(
            _read(store, tmp_path, turn.turn_id, item_id="history", query="needle-at-end").content[0].data["text"]
        )
        assert first["matched"] is False and first["search_complete"] is False
        assert first["next_index"] is not None
        second = json.loads(
            _read(
                store, tmp_path, turn.turn_id, item_id="history", query="needle-at-end", start_index=first["next_index"]
            )
            .content[0]
            .data["text"]
        )
        assert second["matched"] is True and "needle-at-end" in second["text"]
        assert second["history_index"] >= first["next_index"]


def test_runtime_verification_is_in_summary_input_and_searchable_archive(tmp_path: Path):
    from agent_runtime.harness import RolloutContextManager
    from tests.agent.harness.test_compaction_consistency import long_history, start

    with RolloutStore(tmp_path / "verification.db") as store:
        _, turn = start(store, tmp_path)
        _read(store, tmp_path, turn.turn_id, item_id=store.list_items(turn.turn_id)[0].item_id)
        operation = store.list_tool_operations(turn.turn_id)[0]
        verification = store.record_verification(
            turn_id=turn.turn_id,
            operation_id=operation.operation_id,
            kind="inspection",
            verifier="unique-runtime-check-abc",
            verified_resources=("file:ledger.py",),
        )
        long_history(store, turn)
        manager = RolloutContextManager(store)
        source_hash, source = manager.semantic_source(turn.turn_id)
        # Storage identity stays in the archive index, not the semantic working
        # memory. The actual verification and chronological position must remain.
        assert verification.item_id not in source
        assert all("item_id" not in entry for entry in json.loads(source)["history"])
        assert "unique-runtime-check-abc" in source and "file:ledger.py" in source
        manager.commit_compaction(
            manager.semantic_candidate(turn.turn_id, source_hash=source_hash, summary="An inspection was recorded.")
        )
        page = json.loads(
            _read(store, tmp_path, turn.turn_id, item_id="history", query="unique-runtime-check-abc")
            .content[0]
            .data["text"]
        )
        assert page["matched"] is True and page["matched_item_id"] == verification.item_id
