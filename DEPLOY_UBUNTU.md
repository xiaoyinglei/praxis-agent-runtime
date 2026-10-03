# Ubuntu deployment

The Linux command backend uses `/usr/bin/bwrap` (Bubblewrap). Both
`run_command` and `execute_python` support **read-only, network-disabled**
execution. Their private temporary directory is writable. Requests for
`workspace_write=true` or `network=true` return `sandbox_policy_unsupported`
before starting the command. Use the dedicated workspace file tools for edits.
Python computations work; Python-generated workspace artifacts are not yet
supported on Linux. This is not full parity with the macOS backend.

This restriction preserves the existing contract: read-only bind mounts cannot
prevent creation of new reserved `.git`/`.venv` paths in an otherwise writable
workspace, and sharing the host network would expose more than outbound IP
connections. Do not bypass the sandbox with a shell wrapper or silent fallback.

## Host preparation

Use a regular user, Python 3.12 or 3.13, Git, uv, and Bubblewrap:

```bash
sudo apt-get update
sudo apt-get install bubblewrap
```

Ubuntu 24.04 may block user namespaces via AppArmor even with bwrap installed.
Inspect the diagnostic from bwrap and `sudo journalctl -k` first. If logs identify
`unprivileged_userns` denials, an administrator can review and install the
included **per-executable** allowance:

```bash
sudo install -m 0644 scripts/ubuntu/praxis-bwrap.apparmor /etc/apparmor.d/praxis-bwrap
sudo apparmor_parser -r /etc/apparmor.d/praxis-bwrap
```

Do not install a second profile if the host already has a profile attached to
`/usr/bin/bwrap`. This allowance enables user namespaces for that executable;
it does not disable AppArmor globally. Background: [Ubuntu user namespace
restrictions](https://ubuntu.com/blog/ubuntu-23-10-restricted-unprivileged-user-namespaces).

## Repository and dependencies

Deploy the revision containing this Linux change, then run:

```bash
uv sync --frozen
uv run agent --help
uv run pytest -q tests/agent/test_linux_sandbox.py
```

The real Linux tests intentionally fail if bwrap cannot start; installing the
binary alone is insufficient. Fake sandbox fixtures test process mechanics,
not isolation. The Linux backend never falls back to unsandboxed execution.

The current default dependency set also includes RAG, document processing and
local model libraries. On a small server, review download/disk requirements
before syncing the full environment. A manually reduced validation virtualenv
is not equivalent to a complete supported `uv sync` installation.

For noninteractive SSH, uv may not be on PATH even though interactive login
finds it. For the inspected admin account, use `/home/admin/.local/bin/uv` or
explicitly add `$HOME/.local/bin` to PATH in the launch environment.

## Reduced core profile for small servers

For the coding Agent and cloud-model path only, use the checked-in core list
with versions constrained by `uv.lock`. This is a deliberate partial install:
RAG and document/data tools may need additional dependencies, and the full
project metadata still declares those packages. Do not treat this as a full
`uv sync` installation or use `pip check` to claim that all project dependencies
are installed.

```bash
uv export --locked --no-dev --no-hashes --no-emit-project \
  --format requirements-txt --output-file /tmp/praxis-constraints.txt
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -c /tmp/praxis-constraints.txt \
  -r scripts/ubuntu/requirements-core.in pytest
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/agent model probe deepseek-flash --level stream
.venv/bin/agent run "只回复：连接成功" --model deepseek-flash --no-require-workspace-change
.venv/bin/python -m pytest -q tests/agent/test_linux_sandbox.py
.venv/bin/python scripts/agent_delivery_smoke.py --fake-model --verbose
```

Start this reduced installation with `.venv/bin/agent`, not `uv run agent`:
`uv run` normally synchronizes the full project dependency set. Model API keys
belong in the parent Agent environment; sandboxed commands receive a sanitized
environment and do not inherit those keys.

## Model configuration

Inspect model IDs with `uv run agent model list --source`. The checked-in
catalog currently defaults to `openai/gpt-oss-120b` via Groq and reads
`GROQ_API_KEY`. Other configured providers read `DEEPSEEK_API_KEY`,
`MOONSHOT_API_KEY`, or `MIMO_API_KEY`. Set the credential for the model you
actually choose in the server's launch environment; do not commit it or copy
an entire developer home directory. An arbitrary `.env` file is not a promise
that the launcher will load it.

User registrations are separate from the repository, normally at
`~/.config/praxis/models.yaml` (override: `PRAXIS_MODEL_REGISTRY_PATH`). Use the
model registration/probe commands documented in README to verify your endpoint.
Choose a cloud endpoint on a small CPU server; a Mac-local model URL at
`127.0.0.1:8080` points to the Ubuntu host after deployment.

Run the public CLI with the exact selected model and a disposable workspace
before using real project data. A passing tool test is not evidence of a
successful model-backed Agent turn. No inbound HTTP port is needed for CLI use
through SSH. Service supervision is only needed if you later choose an
always-running deployment.

## Independent credential service

For server-only operation without a Mac gateway, see [SERVER_MODELS.md](SERVER_MODELS.md). It installs a separate non-sudo credential service and Agent identity, with keys entered through a hidden terminal prompt. No real provider credentials are included in this repository.
