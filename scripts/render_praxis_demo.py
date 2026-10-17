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

DEMO_BANNER = "Praxis"
FAKE_MODEL_BANNER = "固定模型响应 · 工具与测试真实执行"
TERMINAL_COMMAND = (
    "$ uv run python scripts/agent_delivery_smoke.py "
    "--fake-model --case checkout_demo"
)
FRAME_SIZE = (1200, 760)
BODY_FONT_SIZE = 17
TERMINAL_TEXT_LEFT = 482
TERMINAL_TEXT_RIGHT = 1134
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
            only={"checkout_demo"},
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
    # Present six readable moments from the completed run, rather than racing
    # through internal event logs. No timing or model reasoning is fabricated.
    frames = tuple(_render_frame(frame_texts[-1], stage=stage) for stage in range(6))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[1800, 1800, 2400, 2800, 2200, 4000],
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


def _render_frame(lines: Sequence[str], *, stage: int = 5) -> Image.Image:
    image = Image.new("RGB", FRAME_SIZE, "#101217")
    draw = ImageDraw.Draw(image)
    title_font = _load_ui_font(35)
    label_font = _load_ui_font(17)
    body_font = _load_monospace_font(BODY_FONT_SIZE)
    ui_font = _load_ui_font(23)
    muted, white, orange = "#9ca4b3", "#eeede8", "#edab86"
    green, red = "#9bd4b3", "#f0a0a0"

    draw.text((34, 24), "✳", font=_load_ui_font(34), fill=orange)
    draw.text((78, 26), lines[0], font=_load_monospace_font(28), fill=orange)
    draw.text((34, 76), "从一个问题，到一个验证过的修复。", font=title_font, fill=white)
    draw.text((925, 39), "代码修复 / 执行回放", font=label_font, fill=muted)

    draw.rounded_rectangle((32, 142, 1168, 687), radius=18, fill="#191d25", outline="#303641")
    for x, color in ((53, "#e3827c"), (75, "#debc79"), (97, "#8cbe9c")):
        draw.ellipse((x, 164, x + 10, 174), fill=color)
    draw.text((128, 158), "praxis / checkout", font=body_font, fill=muted)
    draw.text((988, 158), f"0{stage + 1} / 06", font=body_font, fill=orange)
    draw.line((33, 194, 1167, 194), fill="#303641")
    draw.line((434, 195, 434, 686), fill="#303641")

    draw.text((59, 224), "你的任务", font=label_font, fill=orange)
    for y, text in ((267, "100 元打八折应为 80 元。"), (304, "修复计算逻辑，并运行测试。")):
        draw.text((59, y), text, font=ui_font, fill=white)

    labels = ("复现问题", "修改实现", "检查结果")
    details = ("读取代码，运行原有测试", "只修改折扣计算公式", "再次执行全部三项测试")
    completed = (stage >= 2, stage >= 3, stage >= 4)
    active = (stage in (0, 1, 2), stage == 3, stage >= 4)
    for index, (label, detail) in enumerate(zip(labels, details, strict=True)):
        y = 390 + index * 85
        color = green if completed[index] else orange if active[index] else "#596171"
        draw.rounded_rectangle((59, y, 88, y + 29), radius=8, fill="#262d36")
        draw.text((66, y + 3), "✓" if completed[index] else str(index + 1), font=label_font, fill=color)
        draw.text((103, y - 2), label, font=ui_font, fill=white if active[index] or completed[index] else muted)
        draw.text((103, y + 33), detail, font=label_font, fill=muted)

    headings = (
        "先看清问题", "读取 pricing.py", "测试复现了错误",
        "修复百分比换算", "重新运行测试", "修复完成，结果可检查",
    )
    draw.text((474, 223), headings[stage], font=_load_ui_font(27), fill=white)
    draw.text((474, 272), "pricing.py", font=body_font, fill=muted)
    draw.rounded_rectangle((470, 306, 1142, 423), radius=10, fill="#10151c")
    draw.text((482, 323), "def total_after_discount(subtotal, discount_percent):", font=body_font, fill="#c3cfe4")
    if stage < 3:
        draw.text(
            (482, 358), "    return round(subtotal * (1 - discount_percent), 2)",
            font=body_font, fill=red if stage == 2 else white,
        )
    else:
        # The displayed diff comes from the real workspace, not an illustration.
        changed = [
            line.removeprefix("[diff] ") for line in lines
            if line.startswith(("[diff] -    return", "[diff] +    return"))
        ]
        if len(changed) != 2:
            raise RuntimeError("checkout demo must include its actual two-line repair diff")
        for y, text, color in zip((358, 389), changed, (red, green), strict=True):
            draw.text((482, y), _fit_terminal_line(text, font=body_font), font=body_font, fill=color)

    draw.text((474, 457), "$ python3 -m unittest -v", font=body_font, fill=orange)
    if stage < 2:
        draw.text((474, 503), "先复现问题，再开始修改。", font=ui_font, fill=muted)
        draw.text((474, 548), "保留现有接口与测试。", font=label_font, fill=muted)
    elif stage < 4:
        draw.text((474, 502), "修复前：100 元 × 八折 → -1900 元", font=ui_font, fill=red)
        draw.text((474, 550), "FAILED (failures=2)", font=body_font, fill=red)
        draw.text((474, 591), "问题：把百分数 20 直接当成了折扣系数。", font=label_font, fill=muted)
    else:
        for y, label, value in (
            (502, "✓  无折扣", "100 → 100"),
            (540, "✓  八折", "100 → 80"),
            (578, "✓  全额折扣", "100 → 0"),
        ):
            draw.text((474, y), label, font=ui_font, fill=green)
            draw.text((661, y), value, font=ui_font, fill=green)
        draw.text((474, 630), "Ran 3 tests · OK", font=body_font, fill=green)
    draw.text((34, 715), lines[1], font=label_font, fill=muted)
    draw.text((865, 715), "读取 → 复现 → 修改 → 验证", font=label_font, fill=orange)
    return image


def _load_ui_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Prefer fonts with Chinese glyphs when producing the published asset."""
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return _load_monospace_font(size)


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
    line = " ".join(str(value).splitlines()).strip()
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
