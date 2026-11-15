"""skills loader:双根扫描 → frontmatter 解析 → 指纹/去重/排序(设计 §5;3B 笔记 §5)。

本模块只有 loader(T6);T8 的 ``invoke_skill`` 与 T9 的
``materialize_skill_asset`` 会追加在其后的独立小节,请保持这一分区。

扫描顺序(扫描序决定"同一 resolved SKILL.md 首次出现"的胜者):

1. ``$SKILL_PATH``(冒号分隔目录,已存在者)→ :attr:`SkillSource.EXTERNAL`;
2. 从**显式 workspace** 逐级上溯到最近的 ``.git``,每一级取
   ``.agents/skills/`` → :attr:`SkillSource.PROJECT`。

与旧 ``skills/loader.py`` 的有意差异:旧实现从 ``os.getcwd()`` 上溯,praxis
的 **PROJECT walk 与 cwd 无关**——workspace 由调用方传入(CLI resume/
composition 不会因为 cwd 不同得到不同目录)。例外:``$SKILL_PATH`` 里的**相对**
目录仍按进程 cwd ``resolve()``(沿用环境变量旧语义);绝对路径与 ``~`` 展开
不受影响。文件布局与指纹/去重/排序语义逐字移植(旧 loader 在 name 缺失时回退
目录名——brief 要求 name 必填,该回退**不移植**,有意差异):

- ``<root>/<name>/SKILL.md`` 与一层命名空间 ``<root>/<ns>/<name>/SKILL.md``
  (后者 name 前缀 ``<ns>:``);
- frontmatter ``^---\\n…\\n---\\n`` + ``yaml.safe_load`` 必须是 mapping;
  必填 ``name``/``description`` 非空;
- ``unsupported_constraints``(非空 ``allowed_tools``/``paths``)在
  :func:`_build_manifest` **一处**裁决:skip + ``logger.warning``,不进目录;
- fingerprint = ``sha256(yaml.dump(sorted frontmatter) + "\\n" + body)``;
- 按 resolved SKILL.md 路径去重(不是 name);
- 按 ``(SOURCE_PRIORITY, name)`` 稳定排序;
- 单个坏 skill ``log+skip``,绝不炸整个 loader。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from praxis.protocol.json import JsonValue
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.skills import (
    MAX_INVOKE_ARGS_CHARS,
    MAX_SKILL_INSTRUCTIONS_CHARS,
    MAX_SKILL_NAME_CHARS,
    SOURCE_PRIORITY,
    SkillCatalog,
    SkillErrorCode,
    SkillManifest,
    SkillSource,
)
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
    ToolTarget,
    json_schema_input,
)
from praxis.tools.builtins.filesystem import PENDING_TOOL_CALL_ID

logger = logging.getLogger(__name__)

# ── frontmatter 解析 ─────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


class _SkillLoadError(ValueError):
    """单个 SKILL.md 的结构性错误;loader 捕获后 log+skip。"""


def _parse_frontmatter(content: str, skill_file: Path) -> tuple[dict[str, Any], str]:
    """返回 ``(frontmatter, body)``;结构性错误抛 :class:`_SkillLoadError`。"""
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        raise _SkillLoadError(f"{skill_file}: missing YAML frontmatter (--- ... ---)")
    raw = match.group(1)
    body = content[match.end():]
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise _SkillLoadError(f"{skill_file}: invalid YAML frontmatter: {error}") from error
    if not isinstance(loaded, dict):
        raise _SkillLoadError(
            f"{skill_file}: frontmatter must be a YAML mapping, got {type(loaded).__name__}"
        )
    if any(not isinstance(key, str) for key in loaded):
        raise _SkillLoadError(f"{skill_file}: frontmatter keys must be strings")
    return loaded, body


def _required_text(frontmatter: dict[str, Any], field_name: str, skill_file: Path) -> str:
    if field_name not in frontmatter:
        raise _SkillLoadError(f"{skill_file}: missing required field '{field_name}'")
    value = frontmatter[field_name]
    if not isinstance(value, str) or not value.strip():
        raise _SkillLoadError(f"{skill_file}: '{field_name}' must be a non-empty string")
    return value.strip()


def _optional_text(frontmatter: dict[str, Any], field_name: str) -> str | None:
    value = frontmatter.get(field_name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(frontmatter: dict[str, Any], field_name: str, skill_file: Path) -> tuple[str, ...]:
    value = frontmatter.get(field_name)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _SkillLoadError(f"{skill_file}: '{field_name}' must be a list of strings")
    return tuple(str(item).strip() for item in value)


def _boolean_field(frontmatter: dict[str, Any], field_name: str, skill_file: Path) -> bool:
    value = frontmatter.get(field_name)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise _SkillLoadError(f"{skill_file}: '{field_name}' must be a boolean")
    return value


def skill_fingerprint(frontmatter: Mapping[str, JsonValue], body: str) -> str:
    """SKILL.md 的稳定指纹(loader 与激活期重算共用的唯一配方)。

    ``sha256(yaml.dump(sorted frontmatter, sort_keys=True,
    default_flow_style=False) + "\\n" + body)``,UTF-8。

    **激活期重算 procedure(R-T6-1)**:重读 SKILL.md → 解析 frontmatter
    mapping → 置 ``frontmatter["name"] = manifest.name``(清单名已含命名空间
    前缀)→ 调用本函数。结果等于 ``manifest.fingerprint`` 当且仅当文件未变;
    直接对**原始解析**的 frontmatter 重算会让命名空间 skill 误报
    ``skill_content_changed``。仅凭 ``(skill_file, manifest)`` 即可复现。
    """
    canonical = yaml.dump(
        {key: frontmatter[key] for key in sorted(frontmatter)},
        sort_keys=True,
        default_flow_style=False,
    )
    return hashlib.sha256(f"{canonical}\n{body}".encode()).hexdigest()


def _build_manifest(
    frontmatter: dict[str, Any],
    body: str,
    skill_file: Path,
    source: SkillSource,
) -> SkillManifest | None:
    """构造清单;非空约束在这一处 skip+warning 并返回 ``None``。"""
    name = _required_text(frontmatter, "name", skill_file)
    description = _required_text(frontmatter, "description", skill_file)
    allowed_tools = _string_list(frontmatter, "allowed_tools", skill_file)
    path_patterns = _string_list(frontmatter, "paths", skill_file)

    unsupported = tuple(
        field_name
        for field_name, values in (("allowed_tools", allowed_tools), ("paths", path_patterns))
        if values
    )
    if unsupported:
        logger.warning(
            "Skipping skill with unsupported constraints: %s (%s)",
            skill_file,
            ", ".join(unsupported),
        )
        return None

    return SkillManifest(
        name=name,
        description=description,
        when_to_use=_optional_text(frontmatter, "when_to_use"),
        version=_optional_text(frontmatter, "version"),
        skill_file=str(skill_file.resolve()),
        root_dir=str(skill_file.parent.resolve()),
        source=source,
        fingerprint=skill_fingerprint(frontmatter, body),
        disable_model_invocation=_boolean_field(
            frontmatter, "disable_model_invocation", skill_file
        ),
    )


def _load_skill_file(
    skill_file: Path,
    source: SkillSource,
    skill_root: Path,
) -> SkillManifest | None:
    """读并解析单个文件;命名空间前缀在此按相对路径施加。"""
    content = skill_file.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content, skill_file)

    try:
        relative = skill_file.parent.relative_to(skill_root)
        parts = relative.parts
    except ValueError:
        parts = ()
    # parts == () 或 (name,) ⇒ 直接 skill;("ns", "name") ⇒ 一层命名空间。
    if len(parts) >= 2:
        raw_name = frontmatter.get("name")
        if isinstance(raw_name, str):
            frontmatter = dict(frontmatter)
            frontmatter["name"] = f"{parts[0]}:{raw_name}"

    return _build_manifest(frontmatter, body, skill_file, source)


# ── 路径扫描 ─────────────────────────────────────────────────────────


def _scan_skill_dir(root: Path) -> list[Path]:
    """收集 ``<root>/<name>/SKILL.md`` 与一层 ``<root>/<ns>/<name>/SKILL.md``。"""
    if not root.is_dir():
        return []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        logger.warning("Cannot read skills directory: %s", root)
        return []

    skill_files: list[Path] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        direct = entry / "SKILL.md"
        if direct.is_file():
            skill_files.append(direct)
        try:
            subentries = sorted(entry.iterdir())
        except OSError:
            continue
        for subentry in subentries:
            namespaced = subentry / "SKILL.md"
            if subentry.is_dir() and namespaced.is_file():
                skill_files.append(namespaced)
    return skill_files


def _external_roots() -> list[Path]:
    """``$SKILL_PATH`` 冒号分隔目录(expanduser+resolve;不存在/不可用则 log+skip)。

    相对条目按**进程 cwd** ``resolve()``(旧环境变量语义;PROJECT walk 才与
    cwd 无关)。单个毒条目(未知 ``~user`` 的 ``RuntimeError``、resolve 的
    ``OSError``)只跳过该根,绝不炸整个 catalog;重复根只保留首次。
    """
    roots: list[Path] = []
    seen: set[Path] = set()
    for part in os.environ.get("SKILL_PATH", "").split(":"):
        part = part.strip()
        if not part:
            continue
        try:
            path = Path(part).expanduser().resolve()
        except (OSError, RuntimeError):
            logger.warning("Skipping unusable SKILL_PATH entry: %s", part)
            continue
        if not path.is_dir() or path in seen:
            continue
        seen.add(path)
        roots.append(path)
    return roots


def _find_repo_root(workspace: Path) -> Path | None:
    """从 workspace 上溯找最近的 ``.git``(目录或 worktree 的 gitfile)。"""
    current = workspace
    while True:
        if (current / ".git").exists():
            return current
        if current == current.parent:
            return None
        current = current.parent


def _project_roots(workspace: Path) -> list[Path]:
    """workspace → repo root(含端点)每级的 ``.agents/skills/``;无 repo 时仅 workspace。"""
    resolved = workspace.resolve()
    repo_root = _find_repo_root(resolved)
    if repo_root is None:
        return [resolved / ".agents" / "skills"]

    roots: list[Path] = []
    current = resolved
    while True:
        roots.append(current / ".agents" / "skills")
        if current == repo_root:
            return roots
        current = current.parent


# ── 公共 API ─────────────────────────────────────────────────────────


def load_skill_catalog(workspace: Path) -> SkillCatalog:
    """扫描双根并返回按 ``(SOURCE_PRIORITY, name)`` 排序的目录。

    ``workspace`` 是显式传入的装载基点(绝对化后向上找 repo root),**不是
    ``os.getcwd()``**——PROJECT walk 在任何 cwd 下得到同一目录。例外:
    ``$SKILL_PATH`` 的**相对**条目仍按 cwd resolve(见
    :func:`_external_roots`)。单个坏 skill 只 ``log+skip``;重复 resolved 路径
    只保留扫描序首个(``$SKILL_PATH`` 先于 project walk ⇒ 同一物理文件双见时
    记 EXTERNAL)。
    """
    roots: list[tuple[Path, SkillSource]] = [
        (path, SkillSource.EXTERNAL) for path in _external_roots()
    ]
    roots.extend((path, SkillSource.PROJECT) for path in _project_roots(workspace))

    seen: set[Path] = set()
    manifests: list[SkillManifest] = []
    for root, source in roots:
        for skill_file in _scan_skill_dir(root):
            try:
                resolved = skill_file.resolve()
            except OSError:
                logger.warning("Cannot resolve skill path: %s", skill_file)
                continue
            if resolved in seen:
                logger.debug("Skipping duplicate skill: %s", resolved)
                continue
            seen.add(resolved)

            try:
                manifest = _load_skill_file(skill_file, source, root)
            except _SkillLoadError as error:
                logger.warning("Skipping invalid skill: %s", error)
                continue
            except Exception:
                logger.exception("Unexpected error loading skill: %s", skill_file)
                continue
            if manifest is not None:
                manifests.append(manifest)

    manifests.sort(key=lambda manifest: (SOURCE_PRIORITY[manifest.source], manifest.name))
    return SkillCatalog(tuple(manifests))


# ── invoke_skill(T8;T9 的 materialize_skill_asset 追加在本节之后) ──────

_INVOKE_SKILL_NAME = "invoke_skill"

# 旧 integrations/skills.py:39-51 的 InvokeSkillInput(extra=forbid)逐字段移植。
# 有意差异:不复制旧 available_skill_ids 的 name enum——目录已由 T7 注入上下文。
_INVOKE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SKILL_NAME_CHARS,
            "description": "Exact skill name as listed in the skill catalog.",
        },
        "args": {
            "type": ["string", "null"],
            "maxLength": MAX_INVOKE_ARGS_CHARS,
            "description": "Optional value substituted for the skill's $ARGUMENTS placeholder.",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}

_validate_invoke_input = json_schema_input(_INVOKE_INPUT_SCHEMA)

# 旧 skills/runtime.py:14-15:word-boundary 正则 + lambda 替换,替换文本是
# 字面数据(路径/用户参数),绝不被当正则模板解释(无 \1 反引用)。
_SKILL_DIR_RE = re.compile(r"\$SKILL_DIR\b")
_ARGUMENTS_RE = re.compile(r"\$ARGUMENTS\b")


def _activation_error(code: SkillErrorCode, message: str) -> ToolResult:
    """激活失败的固定有界消息;错误结果不带 ``metadata``(T9 只投影成功项)。"""
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name=_INVOKE_SKILL_NAME,
        content=(ToolContentBlock(type="text", data={"text": f"{code}: {message}"}),),
        status=ResultStatus.ERROR,
        is_error=True,
        error_code=code,
        error_message=message,
    )


def _substitute_skill_instructions(instructions: str, root_dir: str, args: str | None) -> str:
    """旧 ``skills/runtime.py:90-93`` 的替换序:字面 ``${SKILL_DIR}`` 先,
    再 word-boundary ``$SKILL_DIR``;``$ARGUMENTS`` 仅在 args 非 None 时替换。"""
    text = instructions.replace("${SKILL_DIR}", root_dir)
    text = _SKILL_DIR_RE.sub(lambda _match: root_dir, text)
    if args is not None:
        text = _ARGUMENTS_RE.sub(lambda _match: args, text)
    return text


def create_invoke_skill_tool(catalog: SkillCatalog) -> Tool:
    """创建 ``invoke_skill``:纯进程内读取,每次激活重读文件重算指纹。

    激活期指纹 procedure(R-T6-1):重读 ``manifest.skill_file`` → 用 loader 的
    同一 parse helpers 解析 → 置 ``frontmatter["name"] = manifest.name``(清单名
    已含命名空间前缀)→ :func:`skill_fingerprint`;不等即 ``skill_content_changed``。
    直接对原始解析重算会让命名空间 skill 误报漂移。

    错误码可达性(R-T8-2):带非空 ``allowed_tools``/``paths`` 的 skill 被 loader
    排除出目录 ⇒ 正常调用是 ``skill_not_found``;``unsupported_skill_constraints``
    只能经"装载后文件被改成带约束"的漂移到达。
    """

    async def run(
        arguments: Mapping[str, JsonValue],
        exec_ctx: ToolExecContext,  # noqa: ARG001 — pure in-process read, no sandbox
    ) -> ToolResult:
        raw_name = arguments.get("name")
        name = raw_name if isinstance(raw_name, str) else ""
        manifest = catalog.get(name)
        if manifest is None:
            return _activation_error(
                "skill_not_found", "skill is not available in the active catalog"
            )
        if manifest.disable_model_invocation:
            return _activation_error(
                "skill_disabled", "skill is not available for model invocation"
            )

        skill_file = Path(manifest.skill_file)
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError:
            return _activation_error("skill_not_found", "skill file is no longer available")

        try:
            frontmatter, body = _parse_frontmatter(content, skill_file)
            # 约束漂移先于指纹比较:R-T8-2 要求该码在"文件已带约束"时可达。
            if _string_list(frontmatter, "allowed_tools", skill_file) or _string_list(
                frontmatter, "paths", skill_file
            ):
                return _activation_error(
                    "unsupported_skill_constraints",
                    "skill now declares unsupported allowed_tools/paths constraints",
                )
        except _SkillLoadError:
            return _activation_error(
                "skill_content_changed", "skill file is no longer a valid SKILL.md"
            )

        frontmatter["name"] = manifest.name
        if skill_fingerprint(frontmatter, body) != manifest.fingerprint:
            return _activation_error(
                "skill_content_changed", "skill content changed since the catalog was loaded"
            )

        raw_args = arguments.get("args")
        args = raw_args if isinstance(raw_args, str) else None
        instructions = _substitute_skill_instructions(body, manifest.root_dir, args)
        truncated = len(instructions) > MAX_SKILL_INSTRUCTIONS_CHARS
        if truncated:
            instructions = instructions[:MAX_SKILL_INSTRUCTIONS_CHARS]

        return ToolResult(
            tool_call_id=PENDING_TOOL_CALL_ID,
            tool_name=_INVOKE_SKILL_NAME,
            content=(
                ToolContentBlock(
                    type="text",
                    data={"text": f"skill activated: {manifest.name}\n{instructions}"},
                ),
            ),
            status=ResultStatus.OK,
            truncated=truncated,
            metadata={
                "skill_id": manifest.name,
                "source": manifest.source.value,
                "fingerprint": manifest.fingerprint,
                "truncated": truncated,
            },
        )

    return Tool(
        definition=ToolDefinition(
            name=_INVOKE_SKILL_NAME,
            description=(
                "Load one skill listed in the skill catalog and return its bounded "
                "instructions. Invoke the matching skill before following that workflow; "
                "never guess an unlisted skill name."
            ),
            input_schema=_INVOKE_INPUT_SCHEMA,
        ),
        validate_input=_validate_invoke_input,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda resolved: ResolvedUse(
            frozenset(),
            (ToolTarget(kind="skill", value=str(resolved.get("name", ""))),),
        ),
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=5.0,
        max_model_output_bytes=300_000,
    )
