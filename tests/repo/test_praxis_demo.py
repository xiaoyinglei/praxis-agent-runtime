from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

REPOSITORY = Path(__file__).parents[2]
CHECKED_DEMO = REPOSITORY / "docs" / "assets" / "praxis-demo.gif"
MAX_GIF_BYTES = 2_000_000


@pytest.fixture
def fake_sandbox_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Reuse the delivery-test sandbox contract on non-macOS CI hosts."""

    from agent_runtime.tools.builtins import shell as shell_module

    executable = tmp_path / "fake-sandbox-exec"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" != "-p" ] || [ "$#" -lt 3 ]; then\n'
        "  exit 64\n"
        "fi\n"
        "shift 2\n"
        'exec "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(shell_module, "_SANDBOX_EXEC_PATH", str(executable))
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def execute_without_seatbelt(
        *argv: str,
        **kwargs: Any,
    ) -> asyncio.subprocess.Process:
        if len(argv) < 5 or argv[0] != str(executable) or argv[1] != "-p":
            raise AssertionError("unexpected sandbox-exec argv")
        return await create_subprocess_exec(*argv[3:], **kwargs)

    monkeypatch.setattr(
        shell_module.asyncio,
        "create_subprocess_exec",
        execute_without_seatbelt,
    )
    return executable


def _load_renderer_module():
    script_path = REPOSITORY / "scripts" / "render_praxis_demo.py"
    assert script_path.is_file(), "Praxis demo renderer script is missing"
    spec = importlib.util.spec_from_file_location(
        "render_praxis_demo",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_each_demo_frame_keeps_both_evidence_labels_visible() -> None:
    module = _load_renderer_module()
    trace = (
        "[inspect] tool:start read_file praxis_demo.py",
        "[inspect] tool:ok read_file",
        "[patch] tool:start apply_patch praxis_demo.py",
        "[patch] tool:ok apply_patch",
        "[verify] tool:start read_file praxis_demo.py",
        "[verify] tool:ok read_file",
        "[complete] status=done answer=praxis demo complete",
    )

    frame_texts = module.build_frame_texts(trace)

    assert len(frame_texts) >= 5
    for lines in frame_texts:
        assert module.DEMO_BANNER in lines
        assert module.FAKE_MODEL_BANNER in lines
    assert trace[-1] in frame_texts[-1]


@pytest.mark.parametrize(
    "absolute_path",
    [
        "/srv/praxis/demo.py",
        r"C:\Users\alice\praxis\demo.py",
        r"\\render-server\workspace\praxis\demo.py",
    ],
)
def test_trace_validation_rejects_absolute_paths(
    absolute_path: str,
) -> None:
    module = _load_renderer_module()

    with pytest.raises(
        ValueError,
        match="absolute local path",
    ):
        module._validate_trace_line(
            f"[inspect] tool:start read_file {absolute_path}"
        )


@pytest.mark.parametrize(
    "relative_text",
    [
        "docs/praxis/demo.py",
        r"docs\praxis\demo.py",
        "C:praxis_demo.py",
        "https://example.test/praxis/demo.py",
    ],
)
def test_trace_validation_keeps_relative_text(relative_text: str) -> None:
    module = _load_renderer_module()
    line = f"[inspect] note={relative_text}"

    assert module._validate_trace_line(line) == line


def test_terminal_trace_fitter_keeps_every_bbox_inside_viewport() -> None:
    module = _load_renderer_module()
    required = (
        "BODY_FONT_SIZE",
        "TERMINAL_TEXT_LEFT",
        "TERMINAL_TEXT_RIGHT",
        "_fit_terminal_line",
    )
    missing = [name for name in required if not hasattr(module, name)]
    assert missing == [], f"pixel-fitting API missing: {missing}"

    trace = tuple(
        f"[{phase}] detail=" + ("relative-segment/" * 80)
        for phase in (
            "inspect",
            "patch",
            "verify",
            "complete",
        )
    )
    frames = module.build_frame_texts(trace)
    font = module._load_monospace_font(module.BODY_FONT_SIZE)
    draw = ImageDraw.Draw(Image.new("RGB", module.FRAME_SIZE))
    elided = 0
    for frame in frames:
        for line in frame[2:]:
            fitted = module._fit_terminal_line(line, font=font)
            bbox = draw.textbbox(
                (module.TERMINAL_TEXT_LEFT, 0),
                fitted,
                font=font,
            )
            assert bbox[2] <= module.TERMINAL_TEXT_RIGHT
            if fitted != line:
                elided += 1
                assert fitted.endswith("...")
    assert elided > 0


@pytest.mark.usefixtures("fake_sandbox_exec")
def test_renderer_replays_public_runtime_into_deterministic_gif(
    tmp_path: Path,
) -> None:
    module = _load_renderer_module()
    first = tmp_path / "first.gif"
    second = tmp_path / "second.gif"

    first_artifact = module.render_demo(first)
    second_artifact = module.render_demo(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_artifact.trace_lines == second_artifact.trace_lines
    assert any("[inspect]" in line for line in first_artifact.trace_lines)
    assert any("[patch]" in line for line in first_artifact.trace_lines)
    assert any("[verify]" in line for line in first_artifact.trace_lines)
    assert first_artifact.trace_lines[-1].startswith("[complete]")
    sanitized = "\n".join(first_artifact.trace_lines)
    assert str(tmp_path) not in sanitized
    assert "/Users/" not in sanitized
    assert "api_key" not in sanitized.casefold()
    _assert_bounded_gif(first)


def test_checked_in_demo_is_animated_and_readme_sized() -> None:
    assert CHECKED_DEMO.is_file(), "checked-in Praxis demo GIF is missing"
    _assert_bounded_gif(CHECKED_DEMO)


def _assert_bounded_gif(path: Path) -> None:
    assert 0 < path.stat().st_size < MAX_GIF_BYTES
    with Image.open(path) as image:
        assert image.format == "GIF"
        assert image.n_frames >= 5
        width, height = image.size
        assert width > 0
        assert height > 0
        sizes = []
        for frame_number in range(image.n_frames):
            image.seek(frame_number)
            sizes.append(image.size)
        assert sizes == [(width, height)] * image.n_frames
