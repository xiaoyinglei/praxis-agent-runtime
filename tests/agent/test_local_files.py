from pathlib import Path
from urllib.parse import quote

import pytest

from agent_runtime.local_files import parse_file_input, resolve_local_file


@pytest.mark.parametrize("style", ["raw", "quoted", "escaped", "uri"])
def test_terminal_file_paths_preserve_spaces_unicode_and_original_location(tmp_path, style):
    source = tmp_path / "客户 资料.txt"
    source.write_text("source")
    value = {"raw": str(source), "quoted": f"'{source}'", "escaped": str(source).replace(" ", "\\ "),
             "uri": "file://" + quote(str(source))}[style]
    assert resolve_local_file(value, base=tmp_path) == source.resolve()
    text, paths = parse_file_input(value, base=tmp_path)
    assert not text
    assert paths == [str(source.resolve())]


def test_file_input_keeps_instructions_and_deduplicates_selected_files(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2")
    text, paths = parse_file_input(f"分析以下文件\n'{source}' '{source}'", base=tmp_path)
    assert text == "分析以下文件"
    assert paths == [str(source.resolve())]


def test_ordinary_text_and_missing_paths_are_not_attachments(tmp_path):
    for value in ("解释 /effort", "don't copy files", "/not/a/file", "https://example.com/file.txt"):
        assert parse_file_input(value, base=tmp_path) == (value, [])
    with pytest.raises(ValueError, match="local"):
        resolve_local_file("file://remotehost/share/file", base=tmp_path)


def test_attachment_is_a_file_not_a_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="文件"):
        resolve_local_file(str(tmp_path), base=tmp_path)
