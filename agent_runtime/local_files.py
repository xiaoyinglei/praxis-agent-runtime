"""Local file references from CLI flags, terminal drops and explicit attachments."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import unquote, urlsplit


def user_path_references(text: str) -> tuple[dict[str, str], ...]:
    """Freeze existing paths supplied directly by the user, without reading contents.

    Mixed prose accepts absolute paths, ~/ paths and local file URIs. Quotes
    or backticks delimit paths containing spaces; no shell expansion is run.
    Never apply this parser to model output, attachments or tool results.
    """
    pattern = (
        r'''[`"']([^`"'\n]+)[`"']|'''
        r'''(?<![\w/:])((?:file://(?:localhost)?/|~/|/)(?:\\ |[^\s`"'<>，。；：！？、,;!?()\[\]{}])+)'''
    )
    candidates = []
    for line in text.splitlines():
        value = line.strip()
        try:
            direct = value.startswith(("/", "~/")) and Path(value).expanduser().exists()
        except (OSError, ValueError):
            direct = False
        if direct:
            candidates.append(value)
        else:
            candidates.extend(match.group(1) or match.group(2) for match in re.finditer(pattern, line))
    references = {}
    for value in candidates:
        if not value.startswith(("/", "~/", "file://")):
            continue
        if value.startswith("file://"):
            uri = urlsplit(value)
            if uri.netloc not in {"", "localhost"} or uri.query or uri.fragment:
                continue
            value = unquote(uri.path)
        value = value.replace("\\ ", " ")
        try:
            path = Path(value).expanduser().absolute()
            resolved = path.resolve()
            kind = "file" if resolved.is_file() else "directory" if resolved.is_dir() else None
        except (OSError, ValueError, RuntimeError):
            continue
        if kind is not None:
            references[str(path)] = {"path": str(path), "resolved_path": str(resolved), "kind": kind}
    return tuple(references.values())


def resolve_local_file(value: str, *, base: Path, resolve_symlinks: bool = True) -> Path:
    """Resolve a local file without copying it or interpreting shell commands."""
    value = value.strip()
    # A native terminal may paste an unquoted path containing spaces.
    direct = Path(value).expanduser()
    direct = direct if direct.is_absolute() else base / direct
    if direct.is_file():
        return direct.resolve() if resolve_symlinks else direct.absolute()
    tokens = shlex.split(value)
    if len(tokens) != 1:
        raise ValueError(f"输入路径不是文件: {value}")
    value = tokens[0]
    if value.startswith("file:"):
        uri = urlsplit(value)
        if uri.scheme != "file" or uri.netloc not in {"", "localhost"} or uri.query or uri.fragment:
            raise ValueError("Only local file:// paths are supported")
        value = unquote(uri.path)
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    if not path.exists():
        raise ValueError(f"输入文件不存在: {value}")
    if not path.is_file():
        raise ValueError(f"输入路径不是文件: {value}")
    return path.resolve() if resolve_symlinks else path.absolute()


def parse_file_input(text: str, *, base: Path) -> tuple[str, list[str]]:
    """Recognize path-only lines, leaving prose (and quoted examples) untouched.

    Dragged files usually arrive as shell-quoted/escaped paths. Explicit /attach
    handles mixed prose or ambiguous formats without guessing at user intent.
    """
    lines: list[str] = []
    files: list[str] = []
    for line in text.splitlines():
        try:
            paths = [resolve_local_file(line, base=base, resolve_symlinks=False)]
        except (OSError, ValueError):
            try:
                tokens = shlex.split(line)
                paths = [resolve_local_file(shlex.quote(token), base=base, resolve_symlinks=False) for token in tokens]
                if not paths:
                    raise ValueError("empty line")
            except (OSError, ValueError):
                lines.append(line)
                continue
        files.extend(str(path) for path in paths)
    return "\n".join(lines).strip(), list(dict.fromkeys(files))
