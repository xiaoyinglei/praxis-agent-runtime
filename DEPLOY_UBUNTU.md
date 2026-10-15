# Ubuntu deployment

The Linux command backend uses `/usr/bin/bwrap` (Bubblewrap). `run_command` and
`execute_python` default to read-only, disconnected execution with writable
private scratch. `network=true` requires policy authorization and command-network
approval, plus `libseccomp.so.2`; its syscall filter blocks Unix IPC, host listeners
and io_uring sockets while permitting outbound IPv4/IPv6. A domain-filtered policy
still refuses arbitrary command networking.

Approved `workspace_write=true` uses a private candidate workspace at the same
absolute working path. A successful command publishes its validated diff only
after protected-path, current-permission, alias and host-version checks. Failed,
timed-out or cancelled commands discard candidate writes. Ordinary input/output
content is bounded to 64 MiB. Protected metadata, dependencies and Agent state
remain read-only; new protected paths cannot be published. File replacement is
atomic, but multi-file publication is not a filesystem transaction. External
subprocess writes remain unsupported. See [permission semantics](docs/permissions.md).

Missing sandbox/filter support or unrepresentable policies fail closed; no shell
wrapper or unrestricted fallback is used. Local stdio MCP remains read-only and
network-disabled on Linux.

## Optional native Codex patch tool

The default file tools continue to work without Codex. To opt into the native
patch format on Linux, install the pinned `rust-v0.162.1` executable **outside
all Agent workspaces**. The standard-library installer supports x86_64 and
aarch64 and verifies the release archive SHA-256 before installing its exact
executable member:

```bash
python3 scripts/ubuntu/install-codex-patch.py --destination "$HOME/.local/lib/praxis/codex-patch"
export PRAXIS_CODEX_APPLY_PATCH="$HOME/.local/lib/praxis/codex-patch"
PRAXIS_TEST_CODEX_BINARY="$PRAXIS_CODEX_APPLY_PATCH" uv run pytest -q tests/agent/test_codex_apply_patch.py tests/agent/test_linux_codex_patch.py
```

`--archive /path/to/downloaded.tar.gz` supports offline installation with the
same digest check. Unset `PRAXIS_CODEX_APPLY_PATCH` to restore the default patch
tool in a direct CLI session. For the isolated `launch-server-agent.py` entrypoint,
install to the same `.local/lib/praxis/codex-patch` path under the `praxis-agent`
account's home. The launcher loads this fixed optional path, ignores ambient
`PRAXIS_CODEX_APPLY_PATCH`, and rejects symlinks, nonexecutables, untrusted owners,
or group/world-writable binaries. Removing or renaming this optional executable
disables it for that launcher. This adapter invokes Codex's offline patch component only; it does not
start Codex's agent or send source code to a model provider. Binary upgrades
require reviewing and updating the pinned archive digests.

Linux runs native Codex inside Bubblewrap with a private view containing only
the declared candidate files. It then validates every output and commits the
changes using the existing atomic file helpers. The child cannot write the
original workspace. Network access is disabled. Protected path components
`.git`, `.venv`, `node_modules`, `.agent_memory`, and `.praxis`, symlink targets
or parents, parent traversal, and targets outside the workspace are rejected.
Each patch has a 30,000-character limit, at most 128 targets, and a 64 MiB total
limit on staged input or output files. Host targets are checked against their
initial contents before commit to detect concurrent edits; this is a trusted
local workspace contract, not a transactional multiwriter filesystem.

Native failures can leave earlier valid edits, as on macOS; the result reports
an error and actual workspace changes. Timeout or cancellation discards the
private view. Invalid output boundaries are rejected before any host commit.
The general command/Python write path uses the separately validated candidate
workspace described above.

## Host preparation

Use a regular user, Python 3.12 or 3.13, Git, uv, and Bubblewrap:

```bash
sudo apt-get update
sudo apt-get install bubblewrap libseccomp2
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
uv sync --locked --link-mode copy
uv run agent --help
uv run pytest -q tests/agent/test_linux_sandbox.py
```

The real Linux tests intentionally fail if bwrap cannot start; installing the
binary alone is insufficient. Fake sandbox fixtures test process mechanics,
not isolation. The Linux backend never falls back to unsandboxed execution.

The default dependency set includes the Agent's document/data tools. RAG
backends and local MLX inference require `--extra rag` and `--extra local-mlx`,
respectively. Use `uv sync --locked --all-extras --link-mode copy` for the full
development environment. A manually reduced validation virtualenv is not
equivalent to a complete supported default `uv sync` installation.

For noninteractive SSH, uv may not be on PATH even though interactive login
finds it. For the inspected admin account, use `/home/admin/.local/bin/uv` or
explicitly add `$HOME/.local/bin` to PATH in the launch environment.

## Reduced core profile for small servers

For the coding Agent and cloud-model path only, use the checked-in core list
with versions constrained by `uv.lock`. This is a deliberate partial install:
Document/data tools need additional dependencies from the default profile;
RAG requires its explicit extra. Do not treat this as a full
`uv sync` installation or use `pip check` to claim that all project dependencies
are installed.

```bash
uv venv --python 3.12 .venv
python3 scripts/ubuntu/install-agent-deps.py . --python .venv/bin/python
# Once this exact lock/profile/Python platform has been installed successfully:
python3 scripts/ubuntu/install-agent-deps.py . --python .venv/bin/python --offline
# Optional test runner, separate from the deployment profile:
uv pip install --python .venv/bin/python --link-mode copy pytest
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

Every core and editable install uses `--link-mode copy --reinstall`. Copy mode
keeps dependency files independent of uv's wheel cache; reinstall also repairs
an existing environment whose installed versions are already correct but whose
files still share cache inodes. Merely setting copy mode on a future install
does not repair those existing files. The installer checks all regular files
in the target venv after installation, including `include/` headers, and fails
if any hardlinks remain. Additional packages outside the core profile must also
be reinstalled with copy mode, or the environment must be recreated.

Prefer keeping the runtime venv outside the Agent workspace for direct CLI
deployments. For example, while in the reviewed repository:

```bash
agent_venv="$HOME/.local/share/praxis/venvs/core"
uv venv --python 3.12 "$agent_venv"
python3 scripts/ubuntu/install-agent-deps.py . --python "$agent_venv/bin/python"
"$agent_venv/bin/agent" run "只回复：连接成功" --model deepseek-flash --no-require-workspace-change
```

The original `.venv` commands remain compatible with the direct credential
launcher and initial model-service installer below. The direct credential
launcher also accepts `--venv /path/to/runtime`, as shown below. A venv left inside the
workspace still participates in the shell containment scan even when the
Agent runs from an external environment; move a retired venv outside the
workspace or repair it before using shell tools.

For an existing reduced `.venv` installation, deploy the reviewed installer and
rerun `python3 scripts/ubuntu/install-agent-deps.py . --python .venv/bin/python
--offline` with the warmed cache. Inspect the final readiness result before
starting the Agent. `workspace_hardlink_detected` means the workspace contains
files that alias another scope, so both `run_command` and `execute_python` refuse
execution before starting a process. This is a deployment problem; do not disable
the check, exclude `.venv`, or grant additional shell permissions to bypass it.

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
`uv run` normally synchronizes the default project dependency set. Model API keys
belong in the parent Agent environment; sandboxed commands receive a sanitized
environment and do not inherit those keys.

## Model configuration

For direct DeepSeek CLI operation without the independent model service:

```bash
.venv/bin/python scripts/ubuntu/start-deepseek-chat.py
# Runtime outside the workspace, using the same private saved credential:
python3 scripts/ubuntu/start-deepseek-chat.py --venv "$HOME/.local/share/praxis/venvs/core"
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

Before loading or prompting for a credential, this launcher runs the installed
shell containment check with the selected venv's Python and a fixed timeout.
An unsafe workspace stops startup with its error code and path; an import or
readiness subprocess failure asks for dependency repair. `--configure` still
explicitly replaces the saved key after readiness succeeds.

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
