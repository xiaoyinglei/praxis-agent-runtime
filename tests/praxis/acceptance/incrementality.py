"""增量性判据的唯一实现:live 冒烟与离线可失败测试共用同一函数。

三条判据按顺序执行,失败原因必须能区分(见 M-2 brief):
1. 内容长度下限——证明 provider 真的产出了多段内容,而不是"一个问题本就
   只有一帧"的短回答;
2. delta 帧数下限——内容够长却只有单帧,提示 provider 批量下发或客户端缓冲;
3. 首个 delta 相对请求起点的比例——全缓冲实现把一切留到最后一帧之后,必然
   落在后 50%,因此这条判据对它是可失败的(旧的 ``delta[0] < finished[-1]``
   对缓冲实现也成立,没有鉴别力)。
"""

from __future__ import annotations

from collections.abc import Sequence

# 一个短回答(如 "ready")也能凑出一帧;用它做下限会把"提示词太短"误判成
# "流式不增量"。40 字符足以覆盖一行以上的真实产出。
MIN_CONTENT_CHARS = 40
FIRST_DELTA_RATIO = 0.5


def incrementality_problem(
    *,
    content_length: int,
    delta_times: Sequence[float],
    started: float,
    finished_at: float,
) -> str | None:
    """返回可读的失败原因;``None`` 表示满足全部三条增量性判据。"""
    if content_length < MIN_CONTENT_CHARS:
        return (
            f"provider 返回内容太短(len={content_length} < {MIN_CONTENT_CHARS}),"
            "无法判定增量性(换更长的提示词)"
        )
    if not delta_times:
        return "no incremental delta event observed"
    if len(delta_times) < 2:
        return "观测到单帧到达，可能是 provider 批量下发或客户端缓冲"
    elapsed = finished_at - started
    if elapsed <= 0:
        return "stream finished immediately; timing measurement is unreliable"
    ratio = (delta_times[0] - started) / elapsed
    if ratio >= FIRST_DELTA_RATIO:
        return (
            f"first delta arrived after {ratio:.0%} of the stream duration "
            f"(>= {FIRST_DELTA_RATIO:.0%}); likely buffered, not incremental"
        )
    return None