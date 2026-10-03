"""Root-only disposable systemd isolation smoke, fake credentials/upstream only.

Usage: sudo python3 -I smoke-model-service.py --wheel-dir /path/to/locked/wheels
Uses a temporary /run tree, a transient DynamicUser unit and private StateDirectory.
Does not install the production unit, identities or real credentials.
"""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(*argv: str) -> None:
    subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("root required for a transient DynamicUser unit")
    source = Path(__file__).resolve().parents[2]
    root = Path(tempfile.mkdtemp(prefix="praxis-model-validation-", dir="/run"))
    root.chmod(0o755)
    unit = root.name
    token = secrets.token_urlsafe(32)
    try:
        run("/usr/bin/python3", "-I", "-m", "venv", str(root / "venv"))
        python = str(root / "venv/bin/python")
        run(
            python,
            "-I",
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "install",
            "--no-index",
            "--find-links",
            str(args.wheel_dir),
            "--require-hashes",
            "--only-binary=:all:",
            "-r",
            str(source / "scripts/ubuntu/requirements-model-service.txt"),
        )
        run(python, "-I", str(source / "scripts/ubuntu/validate-model-venv.py"))
        shutil.copyfile(source / "agent_runtime/server_model_gateway.py", root / "gateway.py")
        credentials = root / "credentials"
        credentials.mkdir(mode=0o700)
        for name, value in {
            "deepseek-key": "fake-deepseek-master-secret",
            "groq-key": "fake-groq-master-secret",
            "agent-token": token,
        }.items():
            path = credentials / name
            path.write_text(value)
            path.chmod(0o400)
        if (
            subprocess.run(
                ["runuser", "-u", "nobody", "--", "test", "-r", str(credentials / "deepseek-key")]
            ).returncode
            == 0
        ):
            raise AssertionError("unrelated user could read provider key")
        (root / "runner.py").write_text(
            """import importlib.util, json, os, sys
from pathlib import Path
import httpx, uvicorn
spec=importlib.util.spec_from_file_location("gateway", Path(__file__).with_name("gateway.py"))
g=importlib.util.module_from_spec(spec); sys.modules["gateway"]=g; spec.loader.exec_module(g)
keys,token=g.load_credentials(["deepseek","groq"])
def reply(req):
    body=json.loads(req.content)
    provider="deepseek" if req.url.host=="api.deepseek.com" else "groq"
    assert req.headers["authorization"]=="Bearer "+keys[provider]
    return httpx.Response(200,json={"model":body["model"],"service_uid":os.getuid(),"provider":provider})
app=g.create_app(keys,token,Path(os.environ["STATE_DIRECTORY"])/"quota.sqlite3",policy=g.ServicePolicy(daily_calls=3),transport=httpx.MockTransport(reply))
uvicorn.run(app,host="127.0.0.1",port=18444,access_log=False,log_level="critical")
"""
        )
        properties = [
            "DynamicUser=yes",
            "ProtectSystem=strict",
            "ProtectHome=yes",
            "PrivateTmp=yes",
            "NoNewPrivileges=yes",
            "PrivateDevices=yes",
            "UMask=0077",
            "LimitCORE=0",
            "RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX",
            "CapabilityBoundingSet=",
            f"StateDirectory={unit}",
            "StateDirectoryMode=0700",
        ]
        properties += [
            f"LoadCredential={name}:{credentials / name}" for name in ("deepseek-key", "groq-key", "agent-token")
        ]
        command = ["systemd-run", f"--unit={unit}", "--service-type=simple", "--quiet"]
        for prop in properties:
            command += ["--property", prop]
        command += [python, "-I", "-B", str(root / "runner.py")]
        run(*command)

        def request(model: str | None = None, authenticated: bool = True) -> tuple[int, dict[str, object]]:
            body = (
                None
                if model is None
                else json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()
            )
            headers = {"Content-Type": "application/json"}
            if authenticated:
                headers["Authorization"] = "Bearer " + token
            path = "/v1/models" if model is None else "/v1/chat/completions"
            req = urllib.request.Request("http://127.0.0.1:18444" + path, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                return exc.code, {}

        for _ in range(100):
            try:
                if request()[0] == 200:
                    break
            except urllib.error.URLError:
                pass
            time.sleep(0.1)
        else:
            raise AssertionError("transient service did not become ready")
        assert request(authenticated=False)[0] == 401
        for model in ("deepseek-flash", "openai/gpt-oss-120b"):
            status, body = request(model)
            assert status == 200 and body["model"] == model and body["service_uid"] != 0
        # Stop/start with the same properties instead of a new state/credential identity.
        run("systemctl", "restart", unit)
        for _ in range(100):
            try:
                if request()[0] == 200:
                    break
            except urllib.error.URLError:
                pass
            time.sleep(0.1)
        assert request("deepseek-flash")[0] == 200
        assert request("deepseek-flash")[0] == 429
        listener = subprocess.check_output(["ss", "-ltnH", "sport = :18444"], text=True)
        assert "127.0.0.1:18444" in listener and "0.0.0.0:18444" not in listener and "[::]:18444" not in listener
        print(
            "Linux systemd smoke passed: non-root credential service, private files, "
            "both routes, loopback auth and restart quota."
        )
    finally:
        subprocess.run(["systemctl", "stop", unit], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["systemctl", "reset-failed", unit], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for state in (Path("/var/lib/private") / unit, Path("/var/lib") / unit):
            if state.is_symlink():
                state.unlink()
            elif state.is_dir():
                shutil.rmtree(state)
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
