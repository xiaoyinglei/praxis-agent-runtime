from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_runtime.memory.compactor import WorkingMemoryCompactor


def test_compacts_old_messages_into_bounded_summary_and_tail() -> None:
    messages = [
        HumanMessage(content=f"message {index}", id=f"h{index}")
        for index in range(5)
    ]

    result = WorkingMemoryCompactor(tail_message_count=2).compact(
        messages,
        now_iso="2026-05-08T00:00:00Z",
    )

    assert result.working_summary is not None
    assert result.working_summary.covered_message_ids == ["h0", "h1", "h2"]
    assert result.working_summary.updated_at == "2026-05-08T00:00:00Z"
    assert result.working_summary.token_count > 0
    assert [message.id for message in result.tail_messages] == ["h3", "h4"]


def test_tail_expands_to_preserve_tool_call_result_pair() -> None:
    messages = [
        HumanMessage(content="find policy", id="h1"),
        AIMessage(
            content="calling tool",
            id="ai1",
            tool_calls=[{"name": "search", "args": {"query": "policy"}, "id": "tc1", "type": "tool_call"}],
        ),
        ToolMessage(content="policy result", id="tool1", tool_call_id="tc1"),
    ]

    result = WorkingMemoryCompactor(tail_message_count=1).compact(messages)

    assert result.working_summary is not None
    assert result.working_summary.covered_message_ids == ["h1"]
    assert [message.id for message in result.tail_messages] == ["ai1", "tool1"]


def test_extracts_only_explicit_working_memory_facts_from_covered_messages() -> None:
    messages = [
        HumanMessage(
            content="alpha",
            id="h1",
            additional_kwargs={
                "working_memory_facts": [
                    {
                        "fact_id": "fact-alpha",
                        "text": "Alpha policy applies to group A.",
                        "evidence_ids": ["ev1"],
                        "confidence": 0.9,
                    }
                ]
            },
        ),
        HumanMessage(content="tail", id="h2"),
    ]

    result = WorkingMemoryCompactor(tail_message_count=1).compact(messages)

    assert len(result.extracted_facts) == 1
    fact = result.extracted_facts[0]
    assert fact.fact_id == "fact-alpha"
    assert fact.text == "Alpha policy applies to group A."
    assert fact.source_message_ids == ["h1"]
    assert fact.evidence_ids == ["ev1"]


def test_plain_text_does_not_create_facts() -> None:
    messages = [
        HumanMessage(content="Alpha policy applies to group A.", id="h1"),
        HumanMessage(content="tail", id="h2"),
    ]

    result = WorkingMemoryCompactor(tail_message_count=1).compact(messages)

    assert result.extracted_facts == []


def test_working_memory_draft_is_json_serializable() -> None:
    messages = [
        HumanMessage(content="old", id="h1"),
        HumanMessage(content="tail", id="h2"),
    ]

    result = WorkingMemoryCompactor(tail_message_count=1).compact(messages)

    dumped = result.model_dump(mode="json")
    assert dumped["working_summary"]["covered_message_ids"] == ["h1"]
    assert dumped["tail_messages"][0]["id"] == "h2"
