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
uv venv --python 3.12 .venv
python3 scripts/ubuntu/install-agent-deps.py . --python .venv/bin/python
# Once this exact lock/profile/Python platform has been installed successfully:
python3 scripts/ubuntu/install-agent-deps.py . --python .venv/bin/python --offline
# Optional test runner, separate from the deployment profile:
uv pip install --python .venv/bin/python pytest
.venv/bin/agent model probe deepseek-flash --level stream
.venv/bin/agent run "只回复：连接成功" --model deepseek-flash --no-require-workspace-change
.venv/bin/python -m pytest -q tests/agent/test_linux_sandbox.py
.venv/bin/python scripts/agent_delivery_smoke.py --fake-model --verbose
```

The installer caches the resolved core manifest by `uv.lock`, profile contents,
and Python ABI/platform. Installation accepts binary wheels whose hashes are in
`uv.lock`, and retains uv's downloaded wheel and editable-build cache under
`~/.cache/praxis-deps/uv`. Use `--cache-dir /path/to/private/cache` to choose a
persistent location; do not prune it between deployments. `--offline` forbids
network access, including build dependencies, and fails clearly if the exact
manifest, compatible wheel, or build backend has not been cached. A new lock or
Python platform needs a new warm-up. The installer never downloads model weights.

If the server's PyPI route is unavailable, download the matching locked Linux
wheels on a trusted machine and transfer them over SSH. Use `--wheel-dir
/path/to/wheels --offline` to install only from that directory with no index.
The same lock hashes are verified; manifest compilation and core installation
use only that directory. A missing, incompatible or corrupt wheel fails
installation. Keep the directory for subsequent deployments. The local project's
editable build additionally needs the build backend cached by a previous online
installation; the core wheelhouse does not contain hatchling. A fresh offline
server must receive that warmed uv build cache as well.

The separate privileged model-service installer retains hashed binary wheels
in `/var/cache/praxis-model-pip`, owned by root. It does not share the Agent's
user-writable cache.

## Bounded Git updates

Use the updater from an already reviewed copy of this repository:

```bash
python3 scripts/ubuntu/update-repo.py /home/admin/praxis --branch main --timeout 60
# Diagnose/fetch without updating working files:
python3 scripts/ubuntu/update-repo.py /home/admin/praxis --branch main --timeout 60 --fetch-only
```

The updater fetches the explicit HTTPS GitHub branch into a temporary ref with
HTTP/1.1 and certificate validation. Only transient network/TLS EOF/timeout
failures are retried, at most three times within the total budget (plus at most
two seconds for local ref cleanup). Authentication, certificate and missing-ref
errors stop immediately. Updates require a clean checkout on the requested
branch and a fast-forward; no stash, reset, force merge or global Git configuration
changes are made. URL rewrites that change the transport are refused.

If the entire GitHub route is unavailable, use an explicitly approved bundle
transferred over SSH, with a full commit ID checked separately:

```bash
python3 scripts/ubuntu/update-repo.py /home/admin/praxis --branch main \
  --bundle /path/to/reviewed.bundle --expected-commit FULL_40_CHARACTER_COMMIT_ID
```

Bundle fallback verifies Git prerequisites and the branch's exact commit before
updating. Commit equality verifies identity, not the bundle's provenance; obtain
the bundle and expected commit from a trusted source. A trusted HTTPS proxy is
another option when direct access is unavailable. Never disable TLS validation,
use an unreviewed GitHub mirror, or increase `http.postBuffer` to address fetch EOF.

Start this reduced installation with `.venv/bin/agent`, not `uv run agent`:
`uv run` normally synchronizes the full project dependency set. Model API keys
belong in the parent Agent environment; sandboxed commands receive a sanitized
environment and do not inherit those keys.

## Model configuration

For direct DeepSeek CLI operation without the independent model service:

```bash
.venv/bin/python scripts/ubuntu/start-deepseek-chat.py
```

The first start prompts for the Key with terminal echo disabled and saves it to
`~/.config/praxis/credentials/deepseek.key` (mode `0600`, directory `0700`, outside
the workspace). Later starts read this file without prompting. Replace the Key
explicitly with `--configure`. The launcher rejects unsafe files and passes the
Key only through the model process environment, never command arguments; existing
tool subprocess environment filtering still applies. A saved Key's format is
checked locally; the provider authenticates each actual API request. This is the
direct single-user CLI path; use the independent credential service below when
separate OS identities are required.

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
