"""Praxis hook 适配器:``hooks.json`` 加载与(command)hook 进程运行器。"""

from praxis.adapters.hooks.config import load_host_hooks
from praxis.adapters.hooks.runner import spawn_hook

__all__ = ["load_host_hooks", "spawn_hook"]
