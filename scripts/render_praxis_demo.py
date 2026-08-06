#!/usr/bin/env python
"""Render the public deterministic Praxis delivery case as a labelled GIF.

The visible content and frame sequence are deterministic. Encoded GIF bytes are
repeatable in the same Pillow and selected-font environment; font fallbacks mean
cross-platform binary identity is intentionally not promised.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from PIL import Image, ImageDraw, ImageFont

DEMO_BANNER = "PRAXIS — DETERMINISTIC DEMO"
FAKE_MODEL_BANNER = "FAKE MODEL — NOT MODEL QUALITY EVIDENCE"
TERMINAL_COMMAND = (
    "$ uv run python scripts/agent_delivery_smoke.py "
    "--fake-model --case praxis_demo"
)
FRAME_SIZE = (960, 540)
BODY_FONT_SIZE = 17
TERMINAL_TEXT_LEFT = 50
TERMINAL_TEXT_RIGHT = 912
MAX_VISIBLE_TRACE_LINES = 12
_MINIMUM_FRAME_COUNT = 5
_SENSITIVE_TRACE_TOKENS = (
    "api_key",
    "authorization:",
    "bearer ",
    "credential",
    "password",
    "secret",
)


@dataclass(frozen=True)
class RenderedDemo:
    trace_lines: tuple[str, ...]
    frame_count: int
    size: tuple[int, int]
    bytes_written: int


def build_frame_texts(
    trace_lines: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Build cumulative terminal frames with evidence labels always present."""

    sanitized = tuple(_validate_trace_line(line) for line in trace_lines)
    if not sanitized:
        raise ValueError("demo trace must contain at least one runtime event")
    frames: list[tuple[str, ...]] = [
        (
            DEMO_BANNER,
            FAKE_MODEL_BANNER,
            TERMINAL_COMMAND,
        )
    ]
    visible: list[str] = []
    for line in sanitized:
        visible.append(line)
        frames.append(
            (
                DEMO_BANNER,
                FAKE_MODEL_BANNER,
                TERMINAL_COMMAND,
                *visible[-MAX_VISIBLE_TRACE_LINES:],
            )
        )
    if len(frames) < _MINIMUM_FRAME_COUNT:
        raise ValueError("demo trace is too short to render an animated proof")
    return tuple(frames)


def render_demo(output: Path) -> RenderedDemo:
    """Execute the fake-model public Agent case and render its real trace."""

    smoke = _load_delivery_smoke()
    results = asyncio.run(
        smoke.run_matrix(
            model="fake",
            fake_model=True,
            only={"praxis_demo"},
        )
    )
    if len(results) != 1:
        raise RuntimeError(
            f"expected one Praxis demo result, received {len(results)}"
        )
    result = results[0]
    if not result.passed:
        raise RuntimeError(f"Praxis demo runtime failed: {result.error}")
    trace_lines = _presentation_trace(
        result.event_lines,
        workspace_diff=result.workspace_diff,
    )
    frame_texts = build_frame_texts(trace_lines)
    frames = tuple(_render_frame(lines) for lines in frame_texts)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[950, *([650] * (len(frames) - 2)), 1400],
        loop=0,
        disposal=2,
        optimize=True,
    )
    return RenderedDemo(
        trace_lines=trace_lines,
        frame_count=len(frames),
        size=FRAME_SIZE,
        bytes_written=output.stat().st_size,
    )


def _presentation_trace(
    event_lines: Sequence[str],
    *,
    workspace_diff: str,
) -> tuple[str, ...]:
    diff_lines = tuple(
        f"[diff] {line}"
        for line in workspace_diff.splitlines()
        if line
    )
    if not diff_lines:
        raise RuntimeError("Praxis demo did not emit a workspace diff")
    trace: list[str] = []
    diff_inserted = False
    for line in event_lines:
        trace.append(line)
        if line == "[patch] tool:ok apply_patch":
            trace.extend(diff_lines)
            diff_inserted = True
    if not diff_inserted:
        raise RuntimeError("Praxis demo trace did not complete apply_patch")
    return tuple(trace)


def _render_frame(lines: Sequence[str]) -> Image.Image:
    image = Image.new("RGB", FRAME_SIZE, "#09111f")
    draw = ImageDraw.Draw(image)
    title_font = _load_monospace_font(25)
    label_font = _load_monospace_font(15)
    body_font = _load_monospace_font(BODY_FONT_SIZE)

    draw.text((42, 28), lines[0], font=title_font, fill="#f5f7ff")
    label_box = draw.textbbox((0, 0), lines[1], font=label_font)
    label_width = label_box[2] - label_box[0]
    draw.rounded_rectangle(
        (42, 70, 66 + label_width, 101),
        radius=8,
        fill="#412b11",
        outline="#f3a83b",
        width=1,
    )
    draw.text((54, 77), lines[1], font=label_font, fill="#ffd58a")

    draw.rounded_rectangle(
        (32, 119, 928, 511),
        radius=14,
        fill="#101827",
        outline="#2a3850",
        width=2,
    )
    for x, color in (
        (53, "#ff6b6b"),
        (75, "#ffd166"),
        (97, "#55d187"),
    ):
        draw.ellipse((x, 137, x + 11, 148), fill=color)
    draw.text(
        (124, 132),
        "praxis-demo · public agent_runtime.Agent facade",
        font=label_font,
        fill="#8291a7",
    )
    draw.line((48, 160, 912, 160), fill="#26354b", width=1)
    draw.text(
        (TERMINAL_TEXT_LEFT, 174),
        _fit_terminal_line(lines[2], font=body_font),
        font=body_font,
        fill="#8bd5ff",
    )

    y = 209
    for line in lines[3:]:
        draw.text(
            (TERMINAL_TEXT_LEFT, y),
            _fit_terminal_line(line, font=body_font),
            font=body_font,
            fill=_trace_color(line),
        )
        y += 23
    return image


def _fit_terminal_line(
    value: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> str:
    """Fit one terminal line to the viewport using the loaded font's pixels."""

    draw = ImageDraw.Draw(Image.new("L", (1, 1)))

    def fits(candidate: str) -> bool:
        bbox = draw.textbbox(
            (TERMINAL_TEXT_LEFT, 0),
            candidate,
            font=font,
        )
        return bbox[2] <= TERMINAL_TEXT_RIGHT

    if fits(value):
        return value

    ellipsis = "..."
    low = 0
    high = len(value)
    while low < high:
        midpoint = (low + high + 1) // 2
        if fits(f"{value[:midpoint]}{ellipsis}"):
            low = midpoint
        else:
            high = midpoint - 1
    return f"{value[:low]}{ellipsis}"


def _trace_color(line: str) -> str:
    if line.startswith("[inspect]"):
        return "#8bd5ff"
    if line.startswith("[patch]"):
        return "#ffd166"
    if line.startswith("[verify]"):
        return "#c4a7ff"
    if line.startswith("[complete]"):
        return "#75e6a4"
    if line.startswith("[diff] +"):
        return "#75e6a4"
    if line.startswith("[diff] -"):
        return "#ff8b8b"
    return "#b7c2d4"


def _load_monospace_font(
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path(__file__).parents[1]
        / "docs"
        / "assets"
        / "DejaVuSansMono.ttf",
        Path("/System/Library/Fonts/Menlo.ttc"),
        Path("/System/Library/Fonts/SFNSMono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _validate_trace_line(value: str) -> str:
    line = " ".join(str(value).split())
    if not line:
        raise ValueError("demo trace contains an empty line")
    smoke = _load_delivery_smoke()
    if smoke._contains_demo_absolute_path(line):
        raise ValueError("demo trace contains an absolute local path")
    lowered = line.casefold()
    if any(token in lowered for token in _SENSITIVE_TRACE_TOKENS):
        raise ValueError("demo trace contains a credential-like value")
    return line[:240]


def _load_delivery_smoke() -> ModuleType:
    module_name = "_agent_delivery_smoke_for_praxis_demo"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    script_path = Path(__file__).with_name("agent_delivery_smoke.py")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load agent_delivery_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination GIF path.",
    )
    args = parser.parse_args()
    artifact = render_demo(args.output)
    print(
        f"wrote {args.output} "
        f"frames={artifact.frame_count} "
        f"size={artifact.size[0]}x{artifact.size[1]} "
        f"bytes={artifact.bytes_written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
