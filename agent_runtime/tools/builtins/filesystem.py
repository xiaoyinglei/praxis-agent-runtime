from __future__ import annotations

import codecs
import difflib
import fnmatch
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_runtime.tools.builtins.file_access import atomic_bytes, file_version, open_binary, workspace_write_lock
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_output,
    pydantic_input,
)
from agent_runtime.workspace import WorkspaceRuntime

_INTERNAL_DIRECTORY = ".agent_memory"
_PROTECTED_VERIFICATION_DIRECTORIES = frozenset(
    {".venv", "node_modules"}
)
_DEFAULT_HIDDEN_DIRECTORIES = frozenset(
    {
        _INTERNAL_DIRECTORY,
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".rag",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_MAX_PATCH_DIFF_CHARS = 12_000
_MAX_PATCH_FILE_BYTES = 64 * 1024 * 1024


class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default=".",
        description="Workspace-relative or authorized absolute directory to list.",
        max_length=4096,
    )
    glob: str | None = Field(
        default=None,
        description="Optional filename glob such as '*.py'.",
        max_length=512,
    )
    limit: int = Field(
        default=200,
        ge=1,
        le=200,
        description="Maximum number of sorted entries to return.",
    )


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    size_bytes: int = Field(ge=0)
    is_directory: bool
    is_symlink: bool


class ListFilesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[FileEntry]
    truncated: bool = False


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        max_length=4096,
        description="Workspace-relative or authorized absolute file path.",
    )
    encoding: str = Field(
        default="utf-8",
        min_length=1,
        max_length=64,
        description="Text encoding used when decoding non-binary bytes.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Zero-based byte offset, including when max_lines is supplied. "
            "Use start_line for a one-based source line instead. "
            "For the next non-overlapping chunk, pass the previous output's "
            "next_offset value."
        ),
    )
    start_line: int | None = Field(
        default=None,
        ge=1,
        description=(
            "One-based source line to start from. Pass search_text.line_number "
            "here, never in offset."
        ),
    )
    max_lines: int | None = Field(
        default=None,
        ge=1,
        le=2_000,
        description=(
            "Maximum lines to read from offset or start_line. When start_line is "
            "set and max_lines is omitted, 200 lines are returned."
        ),
    )
    max_bytes: int = Field(
        default=16_000,
        ge=1,
        le=1_000_000,
        description=(
            "Byte budget from 1 through 1,000,000. UTF-8 pages keep complete characters; "
            "when the budget is smaller than one character, up to four bytes are returned. Omit it to use "
            "the context-safe 16,000-byte default, then advance offset for "
            "another chunk."
        ),
    )

    @model_validator(mode="after")
    def validate_read_mode(self) -> ReadFileInput:
        if self.offset and self.start_line is not None:
            raise ValueError(
                "offset cannot be combined with start_line"
            )
        return self


class ReadFileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    size_bytes: int = Field(ge=0)
    offset: int = Field(ge=0)
    next_offset: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Byte offset for the next non-overlapping chunk, or null when the "
            "end of the file has been reached."
        ),
    )
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    next_line: int | None = Field(
        default=None,
        ge=1,
        description=(
            "One-based start_line for the next non-overlapping line chunk, or "
            "null when line-based reading reached the end of the file."
        ),
    )
    truncated: bool
    is_binary: bool
    binary_format: str | None = None
    recommended_tool: str | None = None
    encoding: str


class ApplyPatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Workspace-relative or authorized absolute path of the file to create or edit.",
    )
    old_string: str | None = Field(
        default=None,
        min_length=1,
        max_length=1_000_000,
        description="Exact text that must already exist; omit only when creating a file.",
    )
    new_string: str = Field(
        max_length=1_000_000,
        description="Replacement text; an empty value deletes the old text.",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence instead of requiring uniqueness.",
    )

    @model_validator(mode="after")
    def validate_operation(self) -> ApplyPatchInput:
        if self.old_string is None and self.replace_all:
            raise ValueError("replace_all requires old_string")
        return self


class ApplyPatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    replaced: bool
    created: bool = False
    occurrences: int = Field(ge=0)
    message: str


@dataclass(frozen=True, slots=True)
class _ApplyPatchRunResult:
    output: ApplyPatchOutput
    diff: str = ""
    diff_truncated: bool = False


_LIST_INPUT_SCHEMA, _validate_list_input = pydantic_input(ListFilesInput)
_LIST_OUTPUT_SCHEMA, _unused_list_output_validator = pydantic_input(ListFilesOutput)
_READ_INPUT_SCHEMA, _validate_read_input = pydantic_input(ReadFileInput)
_READ_OUTPUT_SCHEMA, _unused_read_output_validator = pydantic_input(ReadFileOutput)
_PATCH_INPUT_SCHEMA, _validate_patch_input = pydantic_input(ApplyPatchInput)
_PATCH_OUTPUT_SCHEMA, _unused_patch_output_validator = pydantic_input(ApplyPatchOutput)
_PATCH_ERROR_CODES = {
    "file exists": "file_exists",
    "parent directory not found": "parent_directory_not_found",
    "file not found": "file_not_found",
    "old_string not found": "old_string_not_found",
    "old_string is not unique; set replace_all=true": "old_string_not_unique",
    "replacement produced no change": "patch_no_change",
}


def create_list_files_tool(workspace: WorkspaceRuntime) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name="list_files",
            description=(
                "List one workspace directory in deterministic filename order. "
                "This is not a code-search tool; use it when directory layout or an "
                "unknown filename is the missing fact. Use glob to narrow entries and "
                "limit to bound the result. This tool does not recurse; call it again "
                "for a returned directory."
            ),
            input_schema=_LIST_INPUT_SCHEMA,
        ),
        validate_input=_validate_list_input,
        run=lambda arguments: _list_files(
            workspace,
            ListFilesInput.model_validate(arguments),
        ),
        normalize_output=lambda raw: _normalize_model(
            raw,
            model=ListFilesOutput,
            schema=_LIST_OUTPUT_SCHEMA,
        ),
        output_schema=_LIST_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: _workspace_use(
            workspace,
            str(arguments["path"]),
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
        ),
        execution_revision="builtin-list-files-v2-hidden-generated",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=5.0,
        max_model_output_bytes=300_000,
    )


def create_read_file_tool(workspace: WorkspaceRuntime) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name="read_file",
            description=(
                "Read a bounded byte range from one workspace file. Binary files are "
                "reported without embedding their bytes; supported spreadsheets and "
                "PDFs explicitly route to inspect_data_file. The 16,000-byte default "
                "bounds model context; use offset for later chunks, and never set "
                "max_bytes above 1,000,000. A targeted read after a write can confirm "
                "literal file content. Reuse a sufficient result while the workspace "
                "is unchanged; request another range only for content not yet observed. "
                "A read verifies observed content, not execution behavior. "
                "Binary metadata and a range starting at or "
                "past end-of-file do not verify text content."
            ),
            input_schema=_READ_INPUT_SCHEMA,
        ),
        validate_input=_validate_read_input,
        run=lambda arguments: _read_file(
            workspace,
            ReadFileInput.model_validate(arguments),
        ),
        normalize_output=_normalize_read_file,
        output_schema=_READ_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: _workspace_use(
            workspace,
            str(arguments["path"]),
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
        ),
        execution_revision="builtin-read-file-v4-stable-byte-offset",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=5.0,
        max_model_output_bytes=1_100_000,
    )


def create_apply_patch_tool(workspace: WorkspaceRuntime) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name="apply_patch",
            description=(
                "Create a UTF-8 workspace file, or edit an existing file by exact "
                "text replacement. Omit old_string only when creating a new file. "
                "Without replace_all, the old text must occur exactly once. The write "
                "is atomically installed and creation fails if the path exists. Source and output files are "
                "limited to 64 MiB. Concurrent Praxis edits serialize; an external source-version change "
                "before installation is refused. Verification "
                "toolchains under .venv and node_modules are read-only. A successful "
                "replacement establishes that the supplied text was written; it does "
                "not verify execution behavior. Select verification for the requested "
                "result. If the task supplies the exact file, old text, and new text, "
                "the replacement checks fail closed and can avoid a redundant read. Never reapply the "
                "patch merely to reconfirm it."
            ),
            input_schema=_PATCH_INPUT_SCHEMA,
        ),
        validate_input=_validate_patch_input,
        run=lambda arguments: _apply_patch(
            workspace,
            ApplyPatchInput.model_validate(arguments),
        ),
        normalize_output=_normalize_apply_patch,
        output_schema=_PATCH_OUTPUT_SCHEMA,
        static_effects=frozenset(
            {
                ToolEffect.READ_WORKSPACE,
                ToolEffect.WRITE_WORKSPACE,
            }
        ),
        resolve_use=lambda arguments: _workspace_use(
            workspace,
            str(arguments["file_path"]),
            effects=frozenset(
                {
                    ToolEffect.READ_WORKSPACE,
                    ToolEffect.WRITE_WORKSPACE,
                }
            ),
        ),
        execution_revision="builtin-apply-patch-v3-create-or-replace",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=5.0,
        max_model_output_bytes=50_000,
    )


def _list_files(
    workspace: WorkspaceRuntime,
    request: ListFilesInput,
) -> ListFilesOutput:
    directory = _checked_path(workspace, request.path)
    if not directory.is_dir():
        raise NotADirectoryError(f"workspace directory not found: {request.path}")

    entries: list[FileEntry] = []
    truncated = False
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name in _DEFAULT_HIDDEN_DIRECTORIES:
            continue
        if not workspace.is_readable(path):
            continue
        relative = workspace.path_label(path)
        if request.glob and not (fnmatch.fnmatch(path.name, request.glob) or fnmatch.fnmatch(relative, request.glob)):
            continue
        if len(entries) >= request.limit:
            truncated = True
            break
        metadata = path.lstat()
        is_symlink = path.is_symlink()
        entries.append(
            FileEntry(
                path=relative,
                name=path.name,
                size_bytes=metadata.st_size,
                is_directory=not is_symlink and stat.S_ISDIR(metadata.st_mode),
                is_symlink=is_symlink,
            )
        )
    return ListFilesOutput(entries=entries, truncated=truncated)


def _read_file(
    workspace: WorkspaceRuntime,
    request: ReadFileInput,
) -> ReadFileOutput:
    target = _checked_path(workspace, request.path)
    if not target.is_file():
        raise FileNotFoundError(f"workspace file not found: {request.path}")

    if request.start_line is not None or request.max_lines is not None:
        return _read_file_lines(target, request)

    size = target.stat().st_size
    with open_binary(target) as stream:
        prefix = stream.read(8_192)
        stream.seek(request.offset)
        raw = stream.read(request.max_bytes + 4)
    bounded = _complete_text_page(raw, request.max_bytes, request.encoding)
    truncated = len(raw) > len(bounded)
    binary_format = _binary_format(target, prefix)
    is_binary = binary_format is not None or b"\x00" in bounded
    if is_binary and binary_format is None:
        binary_format = "binary"
    content = "" if is_binary else bounded.decode(request.encoding, errors="replace")
    return ReadFileOutput(
        path=request.path,
        content=content,
        size_bytes=size,
        offset=request.offset,
        next_offset=request.offset + len(bounded) if truncated else None,
        start_line=None,
        end_line=None,
        next_line=None,
        truncated=truncated,
        is_binary=is_binary,
        binary_format=binary_format,
        recommended_tool=(
            "inspect_data_file"
            if binary_format in {"pdf", "xlsx", "xlsm"}
            else None
        ),
        encoding=request.encoding,
    )


def _read_file_lines(target: Path, request: ReadFileInput) -> ReadFileOutput:
    size = target.stat().st_size
    start_line = None if request.offset else (request.start_line or 1)
    max_lines = request.max_lines or 200
    with open_binary(target) as stream:
        prefix = stream.read(8_192)
        binary_format = _binary_format(target, prefix)
        if binary_format is not None or b"\x00" in prefix:
            binary_format = binary_format or "binary"
            return ReadFileOutput(
                path=request.path,
                content="",
                size_bytes=size,
                offset=0,
                next_offset=None,
                start_line=start_line,
                end_line=None,
                next_line=None,
                truncated=False,
                is_binary=True,
                binary_format=binary_format,
                recommended_tool=(
                    "inspect_data_file"
                    if binary_format in {"pdf", "xlsx", "xlsm"}
                    else None
                ),
                encoding=request.encoding,
            )
        stream.seek(request.offset)
        for _line_number in range(1, start_line or 1):
            skipped = stream.readline(65_536)
            while skipped and not skipped.endswith(b"\n"):
                skipped = stream.readline(65_536)
            if not skipped:
                return ReadFileOutput(
                    path=request.path,
                    content="",
                    size_bytes=size,
                    offset=size,
                    next_offset=None,
                    start_line=start_line,
                    end_line=None,
                    next_line=None,
                    truncated=False,
                    is_binary=False,
                    encoding=request.encoding,
                )

        content_offset = stream.tell()
        chunks: list[bytes] = []
        bytes_used = 0
        complete_lines = 0
        byte_limited = False
        while complete_lines < max_lines:
            line_offset = stream.tell()
            remaining = request.max_bytes - bytes_used
            line = stream.readline(max(1, remaining + 4))
            if not line:
                break
            if len(line) > remaining:
                if remaining > 0:
                    page = _complete_text_page(line, remaining, request.encoding)
                    chunks.append(page)
                    bytes_used += len(page)
                    stream.seek(line_offset + len(page))
                else:
                    stream.seek(line_offset)
                byte_limited = True
                break
            chunks.append(line)
            bytes_used += len(line)
            complete_lines += 1

        continuation_offset = stream.tell()
        has_more = bool(stream.read(1))
        line_limited = complete_lines >= max_lines and has_more
        truncated = (byte_limited and has_more) or line_limited
        bounded = b"".join(chunks)
        is_binary = b"\x00" in bounded
        content = "" if is_binary else bounded.decode(request.encoding, errors="replace")
        displayed_lines = complete_lines + (1 if byte_limited and bounded else 0)
        return ReadFileOutput(
            path=request.path,
            content=content,
            size_bytes=size,
            offset=content_offset,
            next_offset=continuation_offset if truncated else None,
            start_line=start_line,
            end_line=(
                None
                if displayed_lines == 0 or start_line is None
                else start_line + displayed_lines - 1
            ),
            next_line=(start_line + complete_lines if line_limited and start_line is not None else None),
            truncated=truncated,
            is_binary=is_binary,
            binary_format="binary" if is_binary else None,
            recommended_tool=None,
            encoding=request.encoding,
        )


def _complete_text_page(raw: bytes, limit: int, encoding: str) -> bytes:
    """Keep UTF-8 continuations attached to their code point, with bounded lookahead."""
    bounded = raw[:limit]
    if codecs.lookup(encoding).name != "utf-8":
        return bounded
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    decoder.decode(bounded, final=False)
    pending = decoder.getstate()[0]
    if not pending:
        return bounded
    complete = len(bounded) - len(pending)
    if complete:
        return bounded[:complete]
    for end in range(len(bounded) + 1, min(len(raw), len(bounded) + 3) + 1):
        decoder.decode(raw[end - 1:end], final=False)
        if not decoder.getstate()[0]:
            return raw[:end]
    return bounded  # Invalid/incomplete source bytes retain the documented replacement decoding.


def _binary_format(path: Path, prefix: bytes) -> str | None:
    suffix = path.suffix.casefold()
    if prefix.startswith(b"%PDF-"):
        return "pdf"
    if prefix.startswith(b"PK\x03\x04"):
        return {
            ".xlsx": "xlsx",
            ".xlsm": "xlsm",
            ".docx": "docx",
            ".pptx": "pptx",
            ".zip": "zip",
        }.get(suffix, "zip")
    if prefix.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole_compound_document"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    return None


def _normalize_read_file(raw: object) -> NormalizedToolOutput:
    validated = ReadFileOutput.model_validate(raw)
    structured = json_schema_output(
        _READ_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    if validated.is_binary:
        recommendation = (
            " Use inspect_data_file on this path."
            if validated.recommended_tool == "inspect_data_file"
            else " Use a tool designed for this binary format."
        )
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code="binary_file_requires_inspection",
            error_message=(
                f"read_file does not decode {validated.binary_format or 'binary'} "
                f"content.{recommendation} Do not retry read_file on the same path."
            ),
            retryable=False,
        )
    return NormalizedToolOutput(structured_content=structured)


def _apply_patch(
    workspace: WorkspaceRuntime,
    request: ApplyPatchInput,
) -> _ApplyPatchRunResult:
    with workspace_write_lock(workspace.root):
        return _apply_patch_locked(workspace, request)


def _apply_patch_locked(workspace: WorkspaceRuntime, request: ApplyPatchInput) -> _ApplyPatchRunResult:
    requested_path = Path(request.file_path)
    if any(
        part.casefold() in _PROTECTED_VERIFICATION_DIRECTORIES
        for part in requested_path.parts
    ):
        raise PermissionError(
            "apply_patch cannot modify the verification toolchain"
        )
    target = _checked_path(workspace, request.file_path, write=True)
    relative = Path(workspace.path_label(target))
    if (
        relative.parts
        and any(
            part.casefold() in _PROTECTED_VERIFICATION_DIRECTORIES
            for part in relative.parts
        )
    ):
        raise PermissionError(
            "apply_patch cannot modify the verification toolchain"
        )
    if not target.exists():
        if request.old_string is not None:
            return _ApplyPatchRunResult(
                ApplyPatchOutput(
                    file_path=request.file_path,
                    replaced=False,
                    occurrences=0,
                    message="file not found",
                )
            )
        if not target.parent.is_dir():
            return _ApplyPatchRunResult(
                ApplyPatchOutput(
                    file_path=request.file_path,
                    replaced=False,
                    occurrences=0,
                    message="parent directory not found",
                )
            )
        diff, diff_truncated = _patch_diff(
            "",
            request.new_string,
            file_path=request.file_path,
        )
        try:
            _atomic_create_text(target, request.new_string)
        except FileExistsError:
            return _ApplyPatchRunResult(
                ApplyPatchOutput(
                    file_path=request.file_path,
                    replaced=False,
                    occurrences=0,
                    message="file exists",
                )
            )
        return _ApplyPatchRunResult(
            ApplyPatchOutput(
                file_path=request.file_path,
                replaced=False,
                created=True,
                occurrences=0,
                message="file created",
            ),
            diff=diff,
            diff_truncated=diff_truncated,
        )
    if request.old_string is None:
        return _ApplyPatchRunResult(
            ApplyPatchOutput(
                file_path=request.file_path,
                replaced=False,
                occurrences=0,
                message="file exists",
            )
        )
    if not target.is_file():
        return _ApplyPatchRunResult(
            ApplyPatchOutput(
                file_path=request.file_path,
                replaced=False,
                occurrences=0,
                message="file not found",
            )
        )

    with open_binary(target) as stream:
        version = file_version(os.fstat(stream.fileno()))
        raw = stream.read(_MAX_PATCH_FILE_BYTES + 1)
        if len(raw) > _MAX_PATCH_FILE_BYTES:
            raise ValueError("apply_patch source exceeds the 64 MiB file limit")
        if file_version(os.fstat(stream.fileno())) != version:
            raise PermissionError("file changed while it was being read")
    current = raw.decode("utf-8")
    occurrences = current.count(request.old_string)
    if occurrences == 0:
        return _ApplyPatchRunResult(
            ApplyPatchOutput(
                file_path=request.file_path,
                replaced=False,
                occurrences=0,
                message="old_string not found",
            )
        )
    if occurrences > 1 and not request.replace_all:
        return _ApplyPatchRunResult(
            ApplyPatchOutput(
                file_path=request.file_path,
                replaced=False,
                occurrences=occurrences,
                message="old_string is not unique; set replace_all=true",
            )
        )

    replacements = occurrences if request.replace_all else 1
    output_bytes = len(raw) + replacements * (len(request.new_string.encode("utf-8"))
                                              - len(request.old_string.encode("utf-8")))
    if output_bytes > _MAX_PATCH_FILE_BYTES:
        raise ValueError("apply_patch output exceeds the 64 MiB file limit")
    updated = (
        current.replace(request.old_string, request.new_string)
        if request.replace_all
        else current.replace(request.old_string, request.new_string, 1)
    )
    if updated == current:
        return _ApplyPatchRunResult(
            ApplyPatchOutput(
                file_path=request.file_path,
                replaced=False,
                occurrences=occurrences if request.replace_all else 1,
                message="replacement produced no change",
            )
        )
    diff, diff_truncated = _patch_diff(
        current,
        updated,
        file_path=request.file_path,
    )
    _atomic_write_text(target, updated, expected=version)
    return _ApplyPatchRunResult(
        ApplyPatchOutput(
            file_path=request.file_path,
            replaced=True,
            occurrences=occurrences if request.replace_all else 1,
            message="patch applied",
        ),
        diff=diff,
        diff_truncated=diff_truncated,
    )


def _patch_diff(
    before: str,
    after: str,
    *,
    file_path: str,
) -> tuple[str, bool]:
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    rendered = "\n".join(lines)
    if len(rendered) <= _MAX_PATCH_DIFF_CHARS:
        return rendered, False
    return (
        f"{rendered[:_MAX_PATCH_DIFF_CHARS]}\n… diff truncated …",
        True,
    )


def _atomic_write_text(path: Path, content: str, *,
                       expected: tuple[int, int, int, int, int] | None = None) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    atomic_bytes(path, content.encode("utf-8"), mode=mode, expected=expected)


def _atomic_create_text(path: Path, content: str) -> None:
    atomic_bytes(path, content.encode("utf-8"), create_only=True)


def _workspace_use(
    workspace: WorkspaceRuntime,
    value: str,
    *,
    effects: frozenset[ToolEffect],
) -> ResolvedToolUse:
    target = workspace.resolve_path(value or ".")
    return ResolvedToolUse(
        effects=effects,
        targets=(ToolTarget(kind="workspace_path", value=str(target)),),
    )


def _checked_path(workspace: WorkspaceRuntime, value: str, *, write: bool = False) -> Path:
    checker = workspace.ensure_writable if write else workspace.ensure_readable
    target = checker(workspace.resolve_path(value or "."))
    relative = Path(workspace.path_label(target))
    if relative.parts and relative.parts[0] == _INTERNAL_DIRECTORY:
        raise PermissionError("agent memory is not exposed as a workspace file")
    return target


def _normalize_model(
    raw: object,
    *,
    model: type[BaseModel],
    schema: Mapping[str, JsonValue],
) -> NormalizedToolOutput:
    validated = model.model_validate(raw)
    structured = json_schema_output(
        schema,
        validated.model_dump(mode="json"),
    )
    return NormalizedToolOutput(structured_content=structured)


def _normalize_apply_patch(raw: object) -> NormalizedToolOutput:
    execution = (
        raw if isinstance(raw, _ApplyPatchRunResult) else _ApplyPatchRunResult(ApplyPatchOutput.model_validate(raw))
    )
    validated = ApplyPatchOutput.model_validate(execution.output)
    structured = json_schema_output(
        _PATCH_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    if validated.replaced or validated.created:
        return NormalizedToolOutput(
            structured_content=structured,
            metadata={
                "file_path": validated.file_path,
                "diff": execution.diff,
                "diff_truncated": execution.diff_truncated,
            },
        )
    return NormalizedToolOutput(
        structured_content=structured,
        is_error=True,
        error_code=_PATCH_ERROR_CODES.get(
            validated.message,
            "patch_not_applied",
        ),
        error_message=validated.message,
        retryable=True,
    )


__all__ = [
    "ApplyPatchInput",
    "ApplyPatchOutput",
    "FileEntry",
    "ListFilesInput",
    "ListFilesOutput",
    "ReadFileInput",
    "ReadFileOutput",
    "create_apply_patch_tool",
    "create_list_files_tool",
    "create_read_file_tool",
]
