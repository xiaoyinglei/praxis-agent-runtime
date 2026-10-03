#!/usr/bin/env bash
set -euo pipefail
set +x
if [[ "$(id -un)" != praxis-agent ]]; then
  echo 'Run this launcher as praxis-agent: sudo -iu praxis-agent praxis-agent chat' >&2
  exit 1
fi
# No credential values in command arguments. Python reads only the proxy token.
exec /usr/bin/env -i PATH=/usr/bin:/bin TERM="${TERM:-dumb}" \
  /opt/praxis/agent-venv/bin/python -I -B /opt/praxis/launch-server-agent.py "$@"
