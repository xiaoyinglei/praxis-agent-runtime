"""通用 Excel 表格结构识别器。

用 header=None 原样读取 sheet，扫描前 N 行，综合多维度信号判断：
- 哪些行是表头（单行或多级）
- 哪些行是说明/标题（应跳过）
- 数据从哪一行开始
- 归一化后的列名

完全不依赖具体列名、文件名或 sheet 名。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd

from rag.ingest.table_semantics import deduplicate_table_columns, normalize_table_column_name

_logger = logging.getLogger("rag.header_detector")

MAX_SCAN_ROWS = 30
_MIN_HEADER_SCORE = 0.55
_MIN_DATA_COLUMNS = 2
_MAX_HEADER_CELL_LENGTH = 80
_CONCISE_HEADER_LENGTH = 15


class HeaderKind(Enum):
    SINGLE = auto()  # 单行表头
    MULTI_LEVEL = auto()  # 多级表头（连续两行以上）
    TITLED = auto()  # 标题行 + 单行表头
    INFERRED = auto()  # 真实表头在数据中（Unnamed 修复）
    NONE = auto()  # 无表头，纯数据


@dataclass(frozen=True, slots=True)
class HeaderDetectionResult:
    header_kind: HeaderKind
    header_row_indices: tuple[int, ...]  # 在原始 sheet 中的行号（0-based）
    data_start_row: int  # 第一行数据在原始 sheet 中的行号
    normalized_columns: list[str]  # 归一化后的列名
    confidence: float  # 0.0 ~ 1.0
    warnings: tuple[str, ...]

    @property
    def column_count(self) -> int:
        return len(self.normalized_columns)


def detect_header(
    df: pd.DataFrame,
    *,
    max_scan_rows: int = MAX_SCAN_ROWS,
) -> HeaderDetectionResult:
    """从 pd.read_excel(..., header=None) 的 DataFrame 中识别表头结构。

    返回 HeaderDetectionResult，低置信度时不建议覆盖列名。
    """
    if df.empty:
        return HeaderDetectionResult(
            header_kind=HeaderKind.NONE,
            header_row_indices=(),
            data_start_row=0,
            normalized_columns=[],
            confidence=0.0,
            warnings=("empty sheet",),
        )

    total_rows, total_cols = df.shape
    scan_rows = min(total_rows, max_scan_rows)
    rows: list[list[str]] = [[_normalize_cell(df.iat[r, c]) for c in range(total_cols)] for r in range(scan_rows)]

    # ── 1. 按行打分 ──
    scores = _score_rows(rows, total_cols)

    # ── 2. 识别表头行簇 ──
    header_indices, kind, confidence, warnings = _detect_header_cluster(scores, rows, total_cols)

    # ── 3. 生成列名 ──
    normalized_columns = _build_columns(header_indices, rows, total_cols, kind)

    # ── 4. 确定数据起始行 ──
    if header_indices:
        data_start = header_indices[-1] + 1
    else:
        data_start = 0

    return HeaderDetectionResult(
        header_kind=kind,
        header_row_indices=header_indices,
        data_start_row=data_start,
        normalized_columns=normalized_columns,
        confidence=confidence,
        warnings=tuple(warnings),
    )


# ═══════════════════════════════════════════════════
# 打分
# ═══════════════════════════════════════════════════


def _normalize_cell(value: object) -> str:
    """NaN/None → 空字符串，其余 str() 后 strip。"""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in ("nan", "none", "null", "na", ""):
            return ""
        return stripped
    return str(value).strip()


def _score_rows(rows: list[list[str]], total_cols: int) -> list[dict[str, float]]:
    """对每一行计算多维信号得分。"""
    results: list[dict[str, float]] = []
    for r_idx, row in enumerate(rows):
        non_empty = [v for v in row if v and v.strip()]
        nc = len(non_empty)

        if nc == 0:
            results.append(
                {
                    "_score": 0.0,
                    "_text_ratio": 0.0,
                    "_unique_ratio": 0.0,
                    "_conciseness": 0.0,
                    "_density": 0.0,
                    "_transition": 0.0,
                    "_numeric_ratio": 0.0,
                }
            )
            continue

        text_count = sum(1 for v in non_empty if _looks_like_text(v))
        numeric_count = nc - text_count
        unique_count = _count_unique_in_context(row, rows, r_idx)
        downstream = max(len(rows) - r_idx - 1, 1)
        avg_len = sum(len(v) for v in non_empty) / nc
        conciseness = (
            1.0 if avg_len <= _CONCISE_HEADER_LENGTH else max(0.0, 1.0 - (avg_len - _CONCISE_HEADER_LENGTH) / 60.0)
        )
        density = nc / max(total_cols, 1)
        transition = _transition_score(rows, r_idx, total_cols)

        text_ratio = text_count / nc
        unique_ratio = unique_count / nc
        numeric_ratio = numeric_count / nc

        # 唯一性信号在少量数据时不可靠（>=5 行才有全权重）
        uniqueness_weight = min(downstream / 5.0, 1.0) * 0.15

        # 数值占比过高是反信号；有任一数字列即不是纯表头行
        numeric_penalty = 0.35 if numeric_ratio > 0.0 else 0.0

        # 密度太低（如 1/5 列有值）→ 更像标题行，不像表头
        density_bonus = 0.0 if density < 0.4 else min(density * 1.2, 1.0) * 0.10

        score = (
            text_ratio * 0.40
            + unique_ratio * uniqueness_weight
            + conciseness * 0.10
            + density_bonus
            + transition * 0.25
            - numeric_penalty
        )

        results.append(
            {
                "_score": score,
                "_text_ratio": text_ratio,
                "_unique_ratio": unique_ratio,
                "_conciseness": conciseness,
                "_density": density,
                "_transition": transition,
                "_numeric_ratio": numeric_ratio,
            }
        )

    return results


def _looks_like_text(value: str) -> bool:
    """数值看起来像文本（列名、标签），而非纯数字。"""
    stripped = value.strip()
    if not stripped:
        return False
    if len(stripped) > _MAX_HEADER_CELL_LENGTH:
        return False
    # 纯数字（含负号、小数点、百分号、科学计数法）
    if re.match(r"^-?[\d,.]+%?$", stripped.replace(" ", "").replace(",", "").replace("，", "")):
        return False
    if re.match(r"^-?\d+\.?\d*[eE][+-]?\d+$", stripped):
        return False
    # 日期格式
    if re.match(r"^\d{1,4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?$", stripped):
        return False
    if re.match(r"^\d+:\d+", stripped):
        return False
    return True


def _count_unique_in_context(row: list[str], all_rows: list[list[str]], current_idx: int) -> int:
    """统计当前行中有多少非空值在后续行中不重复出现。"""
    count = 0
    for col_idx, value in enumerate(row):
        stripped = value.strip()
        if not stripped:
            continue
        is_unique = True
        for later_idx in range(current_idx + 1, len(all_rows)):
            later_value = str(all_rows[later_idx][col_idx]).strip()
            if later_value == stripped:
                is_unique = False
                break
        if is_unique:
            count += 1
    return count


def _transition_score(rows: list[list[str]], current_idx: int, total_cols: int) -> float:
    """下一行看起来和当前行不同（表头→数据过渡信号）。"""
    if current_idx >= len(rows) - 1:
        return 0.0
    this_row = rows[current_idx]
    next_row = rows[current_idx + 1]
    this_text = sum(1 for v in this_row if _looks_like_text(str(v)))
    next_numeric = sum(1 for v in next_row if str(v).strip() and not _looks_like_text(str(v)))
    this_count = sum(1 for v in this_row if str(v).strip())
    next_count = sum(1 for v in next_row if str(v).strip())

    score = 0.0
    if this_text >= max(this_count * 0.5, 2) and next_numeric >= 1:
        score += 0.5
    if this_count > 0 and next_count >= max(this_count * 0.3, 1):
        score += 0.3
    if _row_text_similarity(this_row, next_row) < 0.3:
        score += 0.2
    return min(score, 1.0)


def _row_text_similarity(row_a: list[str], row_b: list[str]) -> float:
    """两行文本相似度（Jaccard-like）。"""
    set_a = {str(v).strip() for v in row_a if str(v).strip()}
    set_b = {str(v).strip() for v in row_b if str(v).strip()}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ═══════════════════════════════════════════════════
# 表头簇检测
# ═══════════════════════════════════════════════════


def _detect_header_cluster(
    scores: list[dict[str, float]],
    rows: list[list[str]],
    total_cols: int,
) -> tuple[tuple[int, ...], HeaderKind, float, list[str]]:
    """从行分数中识别表头簇。"""
    warnings: list[str] = []
    n = len(scores)

    candidates = [
        i
        for i in range(n)
        if scores[i]["_score"] >= _MIN_HEADER_SCORE
        and scores[i]["_text_ratio"] >= 0.5
        and scores[i]["_numeric_ratio"] == 0.0
        and scores[i]["_density"] >= 0.4
        and sum(1 for v in rows[i] if str(v).strip()) >= _MIN_DATA_COLUMNS
    ]

    if not candidates:
        return (), HeaderKind.NONE, 0.3, ["no header-like rows detected"]

    # 找连续簇（中间可间隔 1 行空行）
    clusters = _consecutive_groups_with_gap(candidates, max_gap=1)
    best_cluster = max(clusters, key=lambda c: (len(c), _avg_score(scores, c)))
    cluster_indices = tuple(best_cluster)

    # 判断 kind
    kind = _classify_kind(cluster_indices, scores, rows, best_cluster, n)

    # 置信度
    confidence = _compute_confidence(cluster_indices, scores, rows, total_cols, kind, warnings)

    return cluster_indices, kind, confidence, warnings


def _consecutive_groups(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups: list[list[int]] = [[indices[0]]]
    for i in indices[1:]:
        if i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _consecutive_groups_with_gap(indices: list[int], *, max_gap: int = 1) -> list[list[int]]:
    """允许最多 max_gap 行的间隔，跳过中间的空行/说明行。"""
    if not indices:
        return []
    groups: list[list[int]] = [[indices[0]]]
    for i in indices[1:]:
        gap = i - groups[-1][-1] - 1
        if gap <= max_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _avg_score(scores: list[dict[str, float]], indices: list[int]) -> float:
    return sum(scores[i]["_score"] for i in indices) / max(len(indices), 1)


def _classify_kind(
    indices: tuple[int, ...],
    scores: list[dict[str, float]],
    rows: list[list[str]],
    cluster: list[int],
    total_scanned: int,
) -> HeaderKind:
    # cluster 有空行间隔 → 前面是标题/说明
    has_gaps = (cluster[-1] - cluster[0] + 1) > len(cluster)
    if has_gaps:
        return HeaderKind.TITLED

    if len(indices) == 1:
        idx = indices[0]
        for i in range(idx):
            row = rows[i]
            non_empty = [v for v in row if str(v).strip()]
            if not non_empty:
                continue
            s = scores[i]
            if s["_score"] < _MIN_HEADER_SCORE or s["_density"] < 0.4:
                if any(len(str(v).strip()) > _MAX_HEADER_CELL_LENGTH for v in non_empty):
                    return HeaderKind.TITLED
                if s["_density"] < 0.4 and non_empty:
                    return HeaderKind.TITLED
        if idx > 0 and any(
            scores[j]["_score"] < 0.2 and sum(1 for v in rows[j] if str(v).strip()) > 0 for j in range(idx)
        ):
            return HeaderKind.INFERRED
        return HeaderKind.SINGLE

    if len(indices) >= 2:
        return HeaderKind.MULTI_LEVEL

    return HeaderKind.NONE


def _compute_confidence(
    indices: tuple[int, ...],
    scores: list[dict[str, float]],
    rows: list[list[str]],
    total_cols: int,
    kind: HeaderKind,
    warnings: list[str],
) -> float:
    if not indices or kind == HeaderKind.NONE:
        return 0.3

    base = _avg_score(scores, list(indices))

    # 奖励：表头后有明显的数据行
    data_idx = indices[-1] + 1
    if data_idx < len(rows):
        data_row = rows[data_idx]
        non_empty = [v for v in data_row if str(v).strip()]
        numeric = [v for v in non_empty if not _looks_like_text(str(v))]
        if non_empty and len(numeric) >= len(non_empty) * 0.3:
            base = min(base + 0.15, 1.0)
        else:
            warnings.append("row after header does not look like data")
    else:
        warnings.append("no rows after detected header")

    # 惩罚：表头行之间相似度太高（可能是重复数据行）
    if len(indices) >= 2:
        sim = _row_text_similarity(rows[indices[0]], rows[indices[1]])
        if sim > 0.7:
            base -= 0.2
            warnings.append("consecutive header rows are too similar")

    # 惩罚：表头列数和数据列数不匹配
    header_cols = max(sum(1 for v in rows[i] if str(v).strip()) for i in indices)
    if header_cols < max(total_cols * 0.3, _MIN_DATA_COLUMNS):
        base -= 0.15
        warnings.append("header column count is low")

    return max(0.0, min(base, 1.0))


# ═══════════════════════════════════════════════════
# 列名构建
# ═══════════════════════════════════════════════════


def _build_columns(
    header_indices: tuple[int, ...],
    rows: list[list[str]],
    total_cols: int,
    kind: HeaderKind,
) -> list[str]:
    """从表头行构建归一化列名。"""
    if not header_indices or kind == HeaderKind.NONE:
        return [f"column_{c}" for c in range(total_cols)]

    if kind == HeaderKind.MULTI_LEVEL:
        return _flatten_multi_level(header_indices, rows, total_cols)

    # SINGLE / TITLED / INFERRED: 用最后一行表头
    header_row = rows[header_indices[-1]]
    columns: list[str] = []
    for c in range(total_cols):
        raw = normalize_table_column_name(header_row[c])
        if not raw:
            raw = f"column_{c + 1}"
        columns.append(raw)
    return deduplicate_table_columns(columns)


def _flatten_multi_level(
    indices: tuple[int, ...],
    rows: list[list[str]],
    total_cols: int,
) -> list[str]:
    """多级表头：拼接各级名称，去重。

    例如行0=('', '2026年', '2025年'), 行1=('辅助', '销量', '销量')
    → ('辅助', '2026年_销量', '2025年_销量')
    """
    header_rows = [rows[i] for i in indices]
    columns: list[str] = []
    for c in range(total_cols):
        parts: list[str] = []
        for row in header_rows:
            val = normalize_table_column_name(row[c]) if c < len(row) else ""
            if val and val not in parts:
                parts.append(val)
        col_name = "_".join(parts) if parts else f"column_{c + 1}"
        columns.append(col_name)
    return deduplicate_table_columns(columns)


def _deduplicate_column(name: str, seen: set[str]) -> str:
    if name not in seen:
        return name
    idx = 2
    while f"{name}_{idx}" in seen:
        idx += 1
    return f"{name}_{idx}"


__all__ = [
    "HeaderDetectionResult",
    "HeaderKind",
    "detect_header",
]
