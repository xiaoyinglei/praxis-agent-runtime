"""skill 协议数据:目录清单与错误码(设计 §5;3B 笔记 §5;剪枝移植)。

旧实现有 8 种 ``SkillSource``、``SkillSummary``/``LoadedSkill``/``extra`` 等
多套表示;praxis 3B 只保留 loader 真正产出的两个根——``PROJECT``(workspace
上溯到 repo root 的 ``.agents/skills/``)与 ``EXTERNAL``(``$SKILL_PATH``)——
以及一份不可变清单。被剪掉的面不是"以后再说",而是有意不带走:

- ``SkillManifest.extra``:未知 frontmatter 字段仍参与指纹,但不进目录(drop);
- ``unsupported_constraints`` 作为**装载期裁决**:非空 ``allowed_tools`` /
  ``paths`` 在本 runtime 无法强制,loader 一处 skip + warning(字段不落清单);
- trust 模型(:class:`SkillSource.BUNDLED`/``USER``/``MANAGED``/``MCP``/
  ``PLUGIN`` 与 ``skill_source_untrusted``)随 trust-model 移除一起删除。

``SkillManifest`` 字段面固定为 loader 的产出形状;``skill_file``/``root_dir``
是**绝对路径字符串**(T8 用它重读文件重算指纹,T9 用它落 asset)。

**激活期指纹重算的唯一正确 procedure(R-T6-1)**:重读 SKILL.md → 解析
frontmatter mapping → 置 ``frontmatter["name"] = manifest.name``(清单名已含
命名空间前缀)→ ``skill_fingerprint(frontmatter, body)``;结果等于
``manifest.fingerprint`` 当且仅当文件未变。若直接对**原始解析**的
frontmatter 重算,命名空间 skill(loader 曾把 name 改成 ``<ns>:<raw>``)会
误报 ``skill_content_changed``。仅凭 ``(skill_file, manifest)`` 即可复现。

``fingerprint`` 上限 128 字符(旧 ``SkillActivationEvent.fingerprint`` 的
``Field(max_length=128)``),loader 实际产出 sha256 64 位小写 hex。
``MAX_SKILL_DESCRIPTION_CHARS``/``MAX_INVOKE_ARGS_CHARS``/
``MAX_SKILL_INSTRUCTIONS_CHARS`` 是 T7/T8 消费的预算常量,不在本层截断。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

# ── Skill source ────────────────────────────────────────────────────


class SkillSource(StrEnum):
    """skill 的装载根;数值见 :data:`SOURCE_PRIORITY`(越小越靠前)。"""

    PROJECT = "project"  # workspace 上溯到 repo root 的 .agents/skills/
    EXTERNAL = "external"  # $SKILL_PATH(冒号分隔)


SOURCE_PRIORITY: dict[SkillSource, int] = {
    SkillSource.PROJECT: 0,
    SkillSource.EXTERNAL: 1,
}

# ── 预算常量(T7/T8 消费) ────────────────────────────────────────────

MAX_SKILL_NAME_CHARS = 500
MAX_SKILL_DESCRIPTION_CHARS = 200
MAX_INVOKE_ARGS_CHARS = 4000
MAX_SKILL_INSTRUCTIONS_CHARS = 40_000
# 上下文目录列表(header + 每行 skill + 省略 marker)的字符总预算(T7 消费;
# 设计 §5.14)。超出时只裁整条尾部条目,绝不留半截行。
MAX_CATALOG_LISTING_CHARS = 4000

# ── 错误码(T8 的 invoke_skill 按此映射结果) ────────────────────────

SkillErrorCode = Literal[
    "skill_not_found",
    "skill_disabled",
    "invalid_skill_manifest",
    "skill_content_changed",
    "unsupported_skill_constraints",
]

_MAX_FINGERPRINT_CHARS = 128

# ── Skill manifest ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """单个 SKILL.md 的不可变元数据(loader 的唯一产出形状)。

    ``when_to_use``/``version`` 可缺省(``None``);``skill_file`` 是 SKILL.md
    的绝对路径,``root_dir`` 是其父目录的绝对路径。指纹覆盖 frontmatter
    (键序规范化)与正文,由 loader 计算;T8 激活时按 R-T6-1 procedure
    重算:解析原始 frontmatter 后先置 ``frontmatter["name"] = manifest.name``
    (清单名已含命名空间前缀),再调用 ``skill_fingerprint``;否则命名空间
    skill 会假报 ``skill_content_changed``。
    """

    name: str
    description: str
    when_to_use: str | None
    version: str | None
    skill_file: str
    root_dir: str
    source: SkillSource
    fingerprint: str
    disable_model_invocation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("skill name must be a non-empty string")
        if len(self.name) > MAX_SKILL_NAME_CHARS:
            raise ValueError(f"skill name must be at most {MAX_SKILL_NAME_CHARS} characters")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("skill description must be a non-empty string")
        for field_name, value in (("when_to_use", self.when_to_use), ("version", self.version)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"skill {field_name} must be a non-empty string or None")
        if not isinstance(self.source, SkillSource):
            raise ValueError("skill source must be a SkillSource")
        for field_name, value in (("skill_file", self.skill_file), ("root_dir", self.root_dir)):
            if (
                not isinstance(value, str)
                or not value
                or "\0" in value
                or not Path(value).is_absolute()
            ):
                raise ValueError(f"skill {field_name} must be a non-empty absolute path")
        if not isinstance(self.fingerprint, str) or len(self.fingerprint) > _MAX_FINGERPRINT_CHARS:
            raise ValueError(
                f"skill fingerprint must be a string of at most {_MAX_FINGERPRINT_CHARS} characters"
            )
        if type(self.disable_model_invocation) is not bool:
            raise ValueError("skill disable_model_invocation must be a boolean")


# ── Skill catalog ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    """一次装载的 skill 目录;``skills`` 已按 loader 的确定性顺序排好。

    去重按 **resolved SKILL.md 路径**(不是 name),因此不同根下的同名 skill
    可以同时存在。:meth:`get` 是线性首命中——顺序由 loader 保证确定
    (``(SOURCE_PRIORITY, name)`` 稳定排序),不需要第二份索引。
    """

    skills: tuple[SkillManifest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.skills, tuple):
            object.__setattr__(self, "skills", tuple(self.skills))
        if any(not isinstance(manifest, SkillManifest) for manifest in self.skills):
            raise ValueError("skill catalog must contain SkillManifest entries")

    def get(self, name: str) -> SkillManifest | None:
        """按 name 返回排序后的首个匹配;不存在返回 ``None``。"""
        for manifest in self.skills:
            if manifest.name == name:
                return manifest
        return None
