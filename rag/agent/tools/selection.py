from __future__ import annotations

import inspect
import json
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolTarget,
    _require_exact_bool,
    _thaw_json,
    json_schema_output,
    pydantic_input,
)

FIND_TOOLS_NAME = "find_tools"
MAX_FIND_TOOL_MATCHES = 5
MAX_DISCOVERABLE_TOOLS = 4096

_MAX_QUERY_CHARS = 1000
_MAX_DESCRIPTION_CHARS = 2000
_MAX_METADATA_FIELD_CHARS = 20_000
_MAX_SCHEMA_SEARCH_CHARS = 20_000
_MAX_DOCUMENT_TOKENS = 768
_MAX_MATCHED_TERMS = 32
_TOKEN_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ가-힯]|[a-z0-9]+")
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "me",
        "of",
        "please",
        "the",
        "to",
        "tool",
        "use",
        "with",
    }
)
_SEARCH_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "list_files": (
            "list files directory tree browse workspace",
            "列出文件 文件列表 目录 目录树 浏览工作区",
        ),
        "search_text": (
            "grep search code text regex find content source",
            "搜索文本 搜索代码 查找内容 正则 源码检索",
        ),
        "read_file": (
            "read open inspect file source",
            "读取文件 打开文件 查看源码",
        ),
        "apply_patch": (
            "apply patch edit modify code file",
            "应用补丁 修改文件 编辑代码 改源码",
        ),
        "run_command": (
            "run execute command shell terminal process",
            "运行命令 执行命令 终端 shell 进程",
        ),
        "update_plan": (
            "update plan tasks progress",
            "更新计划 任务进度 步骤",
        ),
        "search_knowledge": (
            "search knowledge documents evidence citations retrieval",
            "查询知识库 检索文档 查资料 证据 引用",
        ),
        "task": (
            "delegate subagent child agent isolated task",
            "委派任务 子代理 子智能体 分派工作",
        ),
        "invoke_skill": (
            "invoke load skill workflow instructions",
            "调用技能 加载技能 工作流 指令",
        ),
        "materialize_skill_asset": (
            "copy skill script reference asset",
            "复制技能脚本 技能资源 参考资料",
        ),
    }
)

type FindToolsRunner = Callable[[str, int], object | Awaitable[object]]


class ToolConfigurationError(ValueError):
    """Invalid installed-name or public-option configuration."""


class ToolSelectionError(ValueError):
    """A deterministic selection or activation constraint failed."""


class ToolSchemaBudgetError(ToolSelectionError):
    error_code = "tool_schema_budget_exceeded"

    def __init__(
        self,
        *,
        required_bytes: int,
        budget_bytes: int,
        selected_names: tuple[str, ...],
    ) -> None:
        self.required_bytes = required_bytes
        self.budget_bytes = budget_bytes
        self.selected_names = selected_names
        names = ", ".join(name[:200] for name in selected_names[:20])
        super().__init__(
            f"tool schema budget exceeded: required {required_bytes} bytes, budget {budget_bytes} bytes for [{names}]"
        )


class ToolActivationError(ToolSelectionError):
    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ResolvedToolOptions:
    resident_names: tuple[str, ...]
    disabled_names: tuple[str, ...]
    allow_discovery_tools: bool
    uses_default_tools: bool


@dataclass(frozen=True, slots=True)
class ToolActivationReduction:
    active_names: tuple[str, ...]
    activated_names: tuple[str, ...]
    trace_metadata: Mapping[str, JsonValue]


class FindToolsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=_MAX_QUERY_CHARS,
        pattern=r".*\S.*",
        description="Capability to find, in natural language.",
    )
    limit: int = Field(
        default=MAX_FIND_TOOL_MATCHES,
        ge=1,
        le=MAX_FIND_TOOL_MATCHES,
        description="Maximum matches to return and propose for activation.",
    )


class FindToolMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(max_length=_MAX_DESCRIPTION_CHARS)
    score: float = Field(ge=0.0)
    matched_terms: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=_MAX_MATCHED_TERMS,
    )


class FindToolsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=_MAX_QUERY_CHARS)
    matches: tuple[FindToolMatch, ...] = Field(
        default_factory=tuple,
        max_length=MAX_FIND_TOOL_MATCHES,
    )
    proposed_activation_names: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=MAX_FIND_TOOL_MATCHES,
    )
    error_code: str | None = Field(default=None, max_length=200)
    error_message: str | None = Field(default=None, max_length=2000)


_FIND_INPUT_SCHEMA, _validate_find_input = pydantic_input(FindToolsInput)
_FIND_OUTPUT_SCHEMA, _unused_find_output_validator = pydantic_input(FindToolsOutput)


def resolve_tool_options(
    registry_snapshot: Mapping[str, Tool],
    *,
    default_resident_names: Sequence[str],
    configured_resident_names: Sequence[str] = (),
    discoverable_names: Sequence[str] | None = None,
    tools: Sequence[str] | None = None,
    disabled_tools: Sequence[str] = (),
    allow_discovery_tools: bool = False,
) -> ResolvedToolOptions:
    """Resolve stable public tool options without reading runtime state."""

    installed_names = _validate_snapshot(registry_snapshot)
    _require_exact_bool(
        allow_discovery_tools,
        field_name="allow_discovery_tools",
    )
    disabled_names = _ordered_unique_names(
        disabled_tools,
        field_name="disabled_tools",
    )
    _require_installed(
        installed_names,
        disabled_names,
        source="disabled tool",
    )
    disabled = set(disabled_names)
    discoverable = (
        tuple(name for name in installed_names if name != FIND_TOOLS_NAME)
        if discoverable_names is None
        else _ordered_unique_names(
            discoverable_names,
            field_name="discoverable_names",
        )
    )
    _require_installed(
        installed_names,
        discoverable,
        source="discoverable tool",
    )
    discoverable_set = set(discoverable)

    requested_names = None if tools is None else _ordered_unique_names(tools, field_name="tools")
    uses_default_tools = requested_names is None or not requested_names

    if not uses_default_tools:
        assert requested_names is not None
        _require_installed(
            installed_names,
            requested_names,
            source="explicit tool",
        )
        resident_names = tuple(name for name in requested_names if name not in disabled)
        if FIND_TOOLS_NAME in resident_names and not allow_discovery_tools:
            raise ToolConfigurationError("find_tools requires allow_discovery_tools=True when explicitly named")
    else:
        default_names = _ordered_unique_names(
            default_resident_names,
            field_name="default_resident_names",
        )
        configured_names = _ordered_unique_names(
            configured_resident_names,
            field_name="configured_resident_names",
        )
        _require_installed(
            installed_names,
            default_names,
            source="resident tool",
        )
        _require_installed(
            installed_names,
            configured_names,
            source="resident tool",
        )
        default_set = set(default_names)
        configured_set = set(configured_names) - default_set
        base_names = (
            *(name for name in installed_names if name in default_set),
            *(name for name in installed_names if name in configured_set),
        )
        resident_names = tuple(name for name in base_names if name not in disabled and name != FIND_TOOLS_NAME)
        base_name_set = set(base_names)
        hidden_names = tuple(
            name
            for name in installed_names
            if name in discoverable_set
            and name not in base_name_set
            and name not in disabled
            and name != FIND_TOOLS_NAME
        )
        if allow_discovery_tools and hidden_names:
            if FIND_TOOLS_NAME not in installed_names:
                raise ToolConfigurationError("discovery is enabled with hidden tools but find_tools is not installed")
            if FIND_TOOLS_NAME not in disabled:
                resident_names = (*resident_names, FIND_TOOLS_NAME)

    return ResolvedToolOptions(
        resident_names=resident_names,
        disabled_names=disabled_names,
        allow_discovery_tools=allow_discovery_tools,
        uses_default_tools=uses_default_tools,
    )


def select_tools(
    registry_snapshot: Mapping[str, Tool],
    *,
    resident_names: Sequence[str],
    active_names: Sequence[str] = (),
    disabled_names: Sequence[str] = (),
    schema_budget: int | None = None,
) -> tuple[Tool, ...]:
    """Return the only deterministic model-visible Tool sequence."""

    installed_names = _validate_snapshot(registry_snapshot)
    residents = _ordered_unique_names(
        resident_names,
        field_name="resident_names",
    )
    active = _ordered_unique_names(active_names, field_name="active_names")
    disabled_ordered = _ordered_unique_names(
        disabled_names,
        field_name="disabled_names",
    )
    _require_installed(installed_names, residents, source="resident tool")
    _require_installed(installed_names, active, source="active tool")
    _require_installed(
        installed_names,
        disabled_ordered,
        source="disabled tool",
    )
    disabled = set(disabled_ordered)

    visible_names: list[str] = []
    seen: set[str] = set()
    for name in (*residents, *active):
        if name in seen or name in disabled:
            continue
        seen.add(name)
        visible_names.append(name)

    selected = tuple(registry_snapshot[name] for name in visible_names)
    _enforce_schema_budget(selected, schema_budget=schema_budget)
    return selected


def reduce_tool_activation(
    registry_snapshot: Mapping[str, Tool],
    *,
    proposed_names: Sequence[str],
    active_names: Sequence[str] = (),
    resident_names: Sequence[str] = (),
    discoverable_names: Sequence[str] | None = None,
    disabled_names: Sequence[str] = (),
    schema_budget: int | None = None,
    max_active_tools: int | None = None,
) -> ToolActivationReduction:
    """Validate one proposal and return a monotonic ordered active-name value."""

    installed_names = _validate_snapshot(registry_snapshot)
    installed = set(installed_names)
    residents = _ordered_unique_names(
        resident_names,
        field_name="resident_names",
    )
    current_active = _ordered_unique_names(
        active_names,
        field_name="active_names",
    )
    proposals = _ordered_unique_names(
        proposed_names,
        field_name="proposed_names",
        retain_duplicates=True,
    )
    disabled_ordered = _ordered_unique_names(
        disabled_names,
        field_name="disabled_names",
    )
    _require_installed(installed_names, residents, source="resident tool")
    _require_installed(installed_names, current_active, source="active tool")
    _require_installed(
        installed_names,
        disabled_ordered,
        source="disabled tool",
    )
    discoverable = (
        None
        if discoverable_names is None
        else _ordered_unique_names(
            discoverable_names,
            field_name="discoverable_names",
        )
    )
    if discoverable is not None:
        _require_installed(
            installed_names,
            discoverable,
            source="discoverable tool",
        )
    discoverable_set = None if discoverable is None else set(discoverable)
    _validate_max_active_tools(max_active_tools)

    disabled = set(disabled_ordered)
    resident_set = set(residents)
    next_active = list(current_active)
    seen = set(current_active)
    activated: list[str] = []
    for name in proposals:
        if name not in installed:
            raise ToolActivationError(
                "unknown_tool_activation",
                f"unknown proposed tool activation: {name[:500]}",
            )
        if name in disabled:
            raise ToolActivationError(
                "tool_activation_disabled",
                f"proposed tool activation is disabled: {name[:500]}",
            )
        if name in resident_set or name in seen:
            continue
        if discoverable_set is not None and name not in discoverable_set:
            raise ToolActivationError(
                "tool_activation_not_discoverable",
                f"proposed tool activation is not discoverable: {name[:500]}",
            )
        seen.add(name)
        next_active.append(name)
        activated.append(name)

    if max_active_tools is not None and len(next_active) > max_active_tools:
        raise ToolActivationError(
            "tool_activation_count_exceeded",
            f"active tool count would be {len(next_active)}, limit is {max_active_tools}; no tools were evicted",
        )

    active_tuple = tuple(next_active)
    select_tools(
        registry_snapshot,
        resident_names=residents,
        active_names=active_tuple,
        disabled_names=disabled_ordered,
        schema_budget=schema_budget,
    )
    activated_tuple = tuple(activated)
    return ToolActivationReduction(
        active_names=active_tuple,
        activated_names=activated_tuple,
        trace_metadata=MappingProxyType(
            {
                "proposed_activation_names": proposals,
                "activated_names": activated_tuple,
                "active_names": active_tuple,
                "active_tool_count": len(active_tuple),
            }
        ),
    )


def find_tools(
    registry_snapshot: Mapping[str, Tool],
    *,
    query: str,
    discoverable_names: Sequence[str] | None = None,
    resident_names: Sequence[str] = (),
    active_names: Sequence[str] = (),
    disabled_names: Sequence[str] = (),
    limit: int = MAX_FIND_TOOL_MATCHES,
    schema_budget: int | None = None,
    max_active_tools: int | None = None,
) -> FindToolsOutput:
    """Search bounded canonical Tool metadata and propose an atomic delta."""

    installed_names = _validate_snapshot(registry_snapshot)
    clean_query = _validate_query(query)
    _validate_limit(limit)
    residents = _ordered_unique_names(
        resident_names,
        field_name="resident_names",
    )
    active = _ordered_unique_names(active_names, field_name="active_names")
    disabled_ordered = _ordered_unique_names(
        disabled_names,
        field_name="disabled_names",
    )
    candidates = (
        tuple(name for name in installed_names if name != FIND_TOOLS_NAME)
        if discoverable_names is None
        else _ordered_unique_names(
            discoverable_names,
            field_name="discoverable_names",
        )
    )
    _require_installed(installed_names, residents, source="resident tool")
    _require_installed(installed_names, active, source="active tool")
    _require_installed(
        installed_names,
        disabled_ordered,
        source="disabled tool",
    )
    _require_installed(
        installed_names,
        candidates,
        source="discoverable tool",
    )
    _validate_max_active_tools(max_active_tools)
    if len(candidates) > MAX_DISCOVERABLE_TOOLS:
        raise ToolConfigurationError(f"discoverable tool count exceeds bounded limit {MAX_DISCOVERABLE_TOOLS}")

    eligible = set(candidates) - set(residents) - set(active) - set(disabled_ordered)
    eligible.discard(FIND_TOOLS_NAME)
    ordered_candidates = tuple(name for name in installed_names if name in eligible)
    matches = _rank_tool_matches(
        registry_snapshot,
        candidate_names=ordered_candidates,
        query=clean_query,
        limit=limit,
    )
    proposed_names = tuple(match.name for match in matches)
    if not proposed_names:
        return FindToolsOutput(query=clean_query)

    try:
        reduce_tool_activation(
            registry_snapshot,
            resident_names=residents,
            active_names=active,
            proposed_names=proposed_names,
            disabled_names=disabled_ordered,
            schema_budget=schema_budget,
            max_active_tools=max_active_tools,
        )
    except ToolSchemaBudgetError as exc:
        return FindToolsOutput(
            query=clean_query,
            matches=matches,
            error_code=exc.error_code,
            error_message=str(exc)[:2000],
        )
    except ToolActivationError as exc:
        return FindToolsOutput(
            query=clean_query,
            matches=matches,
            error_code=exc.error_code,
            error_message=str(exc)[:2000],
        )

    return FindToolsOutput(
        query=clean_query,
        matches=matches,
        proposed_activation_names=proposed_names,
    )


def create_find_tools_tool(
    search: FindToolsRunner,
    *,
    execution_revision: str = "search-v1",
) -> Tool:
    """Adapt a state-reading search closure without owning activation state."""

    if not callable(search):
        raise TypeError("search must be callable")
    if not isinstance(execution_revision, str) or not execution_revision.strip():
        raise ValueError("execution_revision must be non-empty")

    async def run(arguments: Mapping[str, JsonValue]) -> object:
        query = arguments["query"]
        limit = arguments["limit"]
        if not isinstance(query, str):
            raise TypeError("validated find_tools query must be a string")
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("validated find_tools limit must be an integer")
        result = search(query, limit)
        return await result if inspect.isawaitable(result) else result

    return Tool(
        definition=ToolDefinition(
            name=FIND_TOOLS_NAME,
            description=(
                "Search installed but hidden tool metadata for a capability. "
                "Returns at most five deterministic matches and proposed activation "
                "names; activation is applied atomically by the agent runtime."
            ),
            input_schema=_FIND_INPUT_SCHEMA,
        ),
        validate_input=_validate_find_input,
        run=run,
        normalize_output=_normalize_find_tools_output,
        output_schema=_FIND_OUTPUT_SCHEMA,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(ToolTarget(kind="tool_snapshot", value="frozen"),),
        ),
        execution_revision=f"selection-find-tools-v1:{execution_revision}",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=3.0,
        max_model_output_bytes=250_000,
    )


def _normalize_find_tools_output(raw: object) -> NormalizedToolOutput:
    output = FindToolsOutput.model_validate(raw)
    if output.error_code is None and output.error_message is not None:
        raise ValueError("find_tools error_message requires error_code")
    matched_names = tuple(match.name for match in output.matches)
    if len(set(matched_names)) != len(matched_names):
        raise ValueError("find_tools matches must contain unique names")
    proposed_names = output.proposed_activation_names
    if len(set(proposed_names)) != len(proposed_names):
        raise ValueError("find_tools proposed names must be unique")
    proposed_set = set(proposed_names)
    if proposed_names != tuple(name for name in matched_names if name in proposed_set):
        raise ValueError("find_tools proposed names must be shown matches in match order")
    if output.error_code is not None and proposed_names:
        raise ValueError("find_tools error output cannot propose activation")
    structured = json_schema_output(
        _FIND_OUTPUT_SCHEMA,
        output.model_dump(mode="json"),
    )
    is_error = output.error_code is not None
    return NormalizedToolOutput(
        structured_content=structured,
        is_error=is_error,
        error_code=output.error_code,
        error_message=(output.error_message or output.error_code if is_error else None),
        retryable=output.error_code
        in {
            "tool_activation_count_exceeded",
            "tool_schema_budget_exceeded",
        },
        metadata={
            "matched_tool_names": matched_names,
            "proposed_activation_names": output.proposed_activation_names,
        },
    )


def _rank_tool_matches(
    registry_snapshot: Mapping[str, Tool],
    *,
    candidate_names: tuple[str, ...],
    query: str,
    limit: int,
) -> tuple[FindToolMatch, ...]:
    query_tokens = tuple(token for token in _tokenize(query) if token not in _QUERY_STOPWORDS)
    if not query_tokens or not candidate_names:
        return ()

    documents: list[tuple[str, Counter[str], int]] = []
    document_frequency: Counter[str] = Counter()
    for name in candidate_names:
        tokens = _search_tokens(registry_snapshot[name])
        counts = Counter(tokens)
        documents.append((name, counts, len(tokens)))
        for term in set(query_tokens) & counts.keys():
            document_frequency[term] += 1

    document_count = len(documents)
    average_length = sum(length for _name, _counts, length in documents) / document_count
    k1 = 1.2
    b = 0.75
    scored: list[tuple[float, int, FindToolMatch]] = []
    unique_query_terms = tuple(dict.fromkeys(query_tokens))
    normalized_query = " ".join(unique_query_terms)
    for position, (name, counts, length) in enumerate(documents):
        score = 0.0
        matched_terms: list[str] = []
        for term in unique_query_terms:
            frequency = counts.get(term, 0)
            if frequency <= 0:
                continue
            matched_terms.append(term)
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(1.0 + (document_count - frequency_docs + 0.5) / (frequency_docs + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / max(average_length, 1.0))
            score += inverse_frequency * frequency * (k1 + 1.0) / denominator

        normalized_name = name.lower().replace("_", " ")
        if normalized_query and normalized_query in normalized_name:
            score += 2.0
        if score <= 0.0:
            continue
        tool = registry_snapshot[name]
        match = FindToolMatch(
            name=name,
            description=tool.definition.description[:_MAX_DESCRIPTION_CHARS],
            score=round(score, 6),
            matched_terms=tuple(matched_terms[:_MAX_MATCHED_TERMS]),
        )
        scored.append((score, position, match))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in scored[:limit])


def _search_tokens(tool: Tool) -> tuple[str, ...]:
    canonical_name = tool.definition.name
    name = canonical_name[:_MAX_METADATA_FIELD_CHARS]
    source_namespace = " ".join(name.split("__")[:-1])[:_MAX_METADATA_FIELD_CHARS]
    aliases = " ".join(_SEARCH_ALIASES.get(canonical_name, ()))[:_MAX_METADATA_FIELD_CHARS]
    description = tool.definition.description[:_MAX_METADATA_FIELD_CHARS]
    schema_text = _schema_search_text(tool.definition.input_schema)
    weighted_tokens = [
        *_tokenize(name) * 4,
        *_tokenize(source_namespace) * 3,
        *_tokenize(aliases) * 3,
        *_tokenize(description) * 2,
        *_tokenize(schema_text),
    ]
    return tuple(weighted_tokens[:_MAX_DOCUMENT_TOKENS])


def _schema_search_text(schema: Mapping[str, JsonValue]) -> str:
    parts: list[str] = []
    size = 0

    def append(value: object) -> None:
        nonlocal size
        if not isinstance(value, str) or not value or size >= _MAX_SCHEMA_SEARCH_CHARS:
            return
        remaining = _MAX_SCHEMA_SEARCH_CHARS - size
        bounded = value[:remaining]
        parts.append(bounded)
        size += len(bounded) + 1

    def walk(value: JsonValue, *, depth: int) -> None:
        if depth > 8 or size >= _MAX_SCHEMA_SEARCH_CHARS:
            return
        if isinstance(value, Mapping):
            append(value.get("title"))
            append(value.get("description"))
            properties = value.get("properties")
            if isinstance(properties, Mapping):
                for property_name, property_schema in properties.items():
                    if size >= _MAX_SCHEMA_SEARCH_CHARS:
                        break
                    append(property_name)
                    walk(property_schema, depth=depth + 1)
            for key in ("$defs", "definitions"):
                definitions = value.get(key)
                if isinstance(definitions, Mapping):
                    for definition_name, definition in definitions.items():
                        if size >= _MAX_SCHEMA_SEARCH_CHARS:
                            break
                        append(definition_name)
                        walk(definition, depth=depth + 1)
            for key in ("items", "additionalProperties"):
                nested = value.get(key)
                if isinstance(nested, Mapping):
                    walk(nested, depth=depth + 1)
            for key in ("oneOf", "anyOf", "allOf"):
                variants = value.get(key)
                if isinstance(variants, tuple):
                    for variant in variants:
                        if size >= _MAX_SCHEMA_SEARCH_CHARS:
                            break
                        walk(variant, depth=depth + 1)

    walk(schema, depth=0)
    return " ".join(parts)


def _tokenize(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value.lower().replace("_", " ")))


def _enforce_schema_budget(
    tools: tuple[Tool, ...],
    *,
    schema_budget: int | None,
) -> None:
    if schema_budget is None:
        return
    if not isinstance(schema_budget, int) or isinstance(schema_budget, bool) or schema_budget <= 0:
        raise ValueError("schema_budget must be a positive integer or None")
    payload = [
        {
            "name": tool.definition.name,
            "description": tool.definition.description,
            "input_schema": _thaw_json(tool.definition.input_schema),
        }
        for tool in tools
    ]
    required_bytes = len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if required_bytes > schema_budget:
        raise ToolSchemaBudgetError(
            required_bytes=required_bytes,
            budget_bytes=schema_budget,
            selected_names=tuple(tool.definition.name for tool in tools),
        )
def _validate_snapshot(registry_snapshot: Mapping[str, Tool]) -> tuple[str, ...]:
    if not isinstance(registry_snapshot, Mapping):
        raise TypeError("registry_snapshot must be a mapping")
    names: list[str] = []
    for name, tool in registry_snapshot.items():
        if not isinstance(name, str) or not name:
            raise ToolConfigurationError("registry snapshot names must be non-empty strings")
        if not isinstance(tool, Tool):
            raise ToolConfigurationError(f"registry snapshot value is not a Tool: {name}")
        if tool.definition.name != name:
            raise ToolConfigurationError(
                f"registry snapshot key does not match Tool definition name: {name} != {tool.definition.name}"
            )
        names.append(name)
    return tuple(names)


def _ordered_unique_names(
    names: Sequence[str],
    *,
    field_name: str,
    retain_duplicates: bool = False,
) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise TypeError(f"{field_name} must be a sequence of names")
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name:
            raise ToolConfigurationError(f"{field_name} must contain non-empty strings")
        if retain_duplicates or name not in seen:
            ordered.append(name)
        seen.add(name)
    return tuple(ordered)


def _require_installed(
    installed_names: tuple[str, ...],
    requested_names: tuple[str, ...],
    *,
    source: str,
) -> None:
    installed = set(installed_names)
    missing = tuple(name for name in requested_names if name not in installed)
    if missing:
        display = ", ".join(name[:200] for name in missing[:20])
        raise ToolConfigurationError(f"unknown {source} name(s): {display}")


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    clean_query = query.strip()
    if not clean_query:
        raise ToolConfigurationError("query must not be blank")
    if len(clean_query) > _MAX_QUERY_CHARS:
        raise ToolConfigurationError(f"query exceeds {_MAX_QUERY_CHARS} characters")
    return clean_query


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise TypeError("limit must be an integer")
    if limit < 1 or limit > MAX_FIND_TOOL_MATCHES:
        raise ValueError(f"limit must be between 1 and {MAX_FIND_TOOL_MATCHES}")


def _validate_max_active_tools(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_active_tools must be a positive integer or None")
__all__ = [
    "FIND_TOOLS_NAME",
    "MAX_DISCOVERABLE_TOOLS",
    "MAX_FIND_TOOL_MATCHES",
    "FindToolMatch",
    "FindToolsInput",
    "FindToolsOutput",
    "FindToolsRunner",
    "ResolvedToolOptions",
    "ToolActivationError",
    "ToolActivationReduction",
    "ToolConfigurationError",
    "ToolSchemaBudgetError",
    "ToolSelectionError",
    "create_find_tools_tool",
    "find_tools",
    "reduce_tool_activation",
    "resolve_tool_options",
    "select_tools",
]
