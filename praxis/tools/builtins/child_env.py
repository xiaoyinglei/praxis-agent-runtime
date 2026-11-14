"""子进程环境的唯一构造入口(R-T16-2;findings S7 同族)。

``bash`` 与 ``run_python`` 派生的一切子进程都必须使用 ``child_env``:
宿主环境变量一律**不继承**。这是"凭据不进沙箱"的结构保证——Praxis 进程
自身持有模型 API key(见 ``api_key_env``),若把 ``os.environ`` 透传给
``/bin/bash`` 或 Python 脚本,``printenv``/``os.environ`` 就能把密钥写进
tool_result、item 日志与模型上下文,文件层的地板/私有目录 deny 随即被绕过。

白名单语义(**有意最小化**,不是旧 ``_command_environment`` 的逐字移植):
- ``PATH`` 为固定列表(工作区 venv/node_modules → 常见系统工具链目录),
  **不读取宿主 PATH**;
- ``HOME``/``TMPDIR``/``TEMP``/``TMP`` 全部指向 ``scratch_dir``——沙箱内
  唯一可写临时区(无沙箱的单测路径用进程私有临时目录,见
  ``scratch_dir_for``);
- ``GIT_CONFIG_GLOBAL=/dev/null``、``GIT_CONFIG_NOSYSTEM=1``:git 读不到
  宿主全局/系统配置;``LC_ALL=C`` 固定输出编码语义。

已知取舍(与旧沙箱 bash 相同,不是回归):旧 ``_command_environment`` 还会
传 ``LANG``/``PIP_CACHE_DIR``/``PYTHONDONTWRITEBYTECODE``/``SHELL``/
``UV_CACHE_DIR``/``XDG_CACHE_HOME``;这里一律不带。因此 ``uv``/``cargo`` 等
装在用户目录的工具在沙箱内通常不可解析——需要时用 ``run_python``/绝对路径,
或由运维把工作区工具链放进 ``.venv/bin``。

除此之外没有任何键。本模块从不读取 ``os.environ``——"最小环境"不是调用方
的选择,而是构造出的字典必然缺少宿主变量这一事实。
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import threading
from pathlib import Path

from praxis.protocol.sandbox import SandboxSpec

_FIXED_PATH_TAIL = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
    "/Library/Developer/CommandLineTools/usr/bin",
)

_fallback_scratch: Path | None = None
_fallback_lock = threading.Lock()


def child_env(scratch_dir: Path, workspace: Path) -> dict[str, str]:
    """构造子进程的最小环境(全仓唯一实现)。

    返回的字典只含固定白名单键;宿主 ``os.environ`` 中的任何变量
    (包括凭据类变量)都不可能出现在子进程里。``scratch_dir`` 必须是
    沙箱内可写目录(或 ``scratch_dir_for`` 提供的进程私有临时目录)。
    """
    scratch = os.path.realpath(scratch_dir)
    root = os.path.realpath(workspace)
    path = os.pathsep.join(
        (
            os.path.join(root, ".venv", "bin"),
            os.path.join(root, "node_modules", ".bin"),
            *_FIXED_PATH_TAIL,
        )
    )
    return {
        "PATH": path,
        "HOME": scratch,
        "TMPDIR": scratch,
        "TEMP": scratch,
        "TMP": scratch,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "LC_ALL": "C",
    }


def scratch_dir_for(spec: SandboxSpec | None) -> Path:
    """执行可写临时区:沙箱有编译出的 scratch;无沙箱时用进程私有目录。

    单测直连路径(``ToolExecContext.sandbox is None``)也构造同一份最小
    环境,不存在"无沙箱时透传 os.environ"的第二条路径。进程私有目录在
    进程退出时清理,避免每次调用泄漏宿主临时目录。
    """
    if spec is not None:
        return Path(spec.scratch_dir)
    global _fallback_scratch
    with _fallback_lock:
        if _fallback_scratch is None:
            _fallback_scratch = Path(tempfile.mkdtemp(prefix="praxis-child-env-"))
            atexit.register(shutil.rmtree, _fallback_scratch, ignore_errors=True)
        return _fallback_scratch
