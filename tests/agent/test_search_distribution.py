from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


def test_built_wheel_search_runs_outside_repo_with_empty_path(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel, = dist.glob("*.whl")
    installed = tmp_path / "site-packages"
    with zipfile.ZipFile(wheel) as archive:
        metadata, = (name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        assert "Requires-Dist: ripgrep-bin" in archive.read(metadata).decode()
        archive.extractall(installed)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "example.py").write_text("needle = 1\n")
    script = textwrap.dedent(
        """
        import asyncio
        import json
        from pathlib import Path
        import agent_runtime
        from agent_runtime.tools.builtins.search import create_search_text_tool
        from agent_runtime.workspace import open_workspace

        async def main():
            tool = create_search_text_tool(open_workspace(Path.cwd()))
            result = await tool.run({"pattern": "needle", "path": "example.py", "regex": False})
            print(json.dumps({"package": agent_runtime.__file__, "result": result.model_dump()}))

        asyncio.run(main())
        """
    )
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(installed), PATH="")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["package"].startswith(str(installed))
    assert payload["result"]["matches"][0]["line_content"] == "needle = 1"
    assert payload["result"]["searched_file_path"] == "example.py"
    assert payload["result"]["truncated"] is False
