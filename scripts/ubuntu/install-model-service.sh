#!/usr/bin/env bash
# First installation only. Reviewed source and a working Linux Agent venv required.
set -euo pipefail
set +x
if [[ "$EUID" != 0 ]]; then
  echo 'Run the reviewed installer with sudo on Ubuntu.' >&2
  exit 1
fi
source_root="$(cd -- "${1:?Usage: install-model-service.sh /path/to/reviewed/praxis}" && pwd -P)"
model_cache=/var/cache/praxis-model-pip
# The privileged installer never consumes an Agent/user-writable wheel cache.
/usr/bin/python3 -I - "$model_cache" <<'PY'
import os
import stat
import sys
from pathlib import Path
cache = Path(sys.argv[1])
if cache.is_symlink():
    raise SystemExit('Model wheel cache must not be a symlink.')
if cache.exists():
    info = cache.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise SystemExit('Model wheel cache must be root-owned and not writable by other users.')
PY
for destination in /opt/praxis /etc/praxis-model /etc/systemd/system/praxis-model.service /usr/local/bin/praxis-agent /usr/local/bin/praxis /var/lib/praxis-model /var/lib/praxis-agent /var/lib/private/praxis-model; do
  if [[ -e "$destination" || -L "$destination" ]]; then
    echo "Initial installation refuses existing destination: $destination" >&2
    exit 1
  fi
done
for account in praxis-model praxis-agent; do
  if getent passwd "$account" >/dev/null || getent group "$account" >/dev/null; then
    echo "Initial installation refuses existing identity: $account" >&2
    exit 1
  fi
done
/usr/bin/python3 -I "$source_root/scripts/ubuntu/validate-model-venv.py" \
  --agent-config "$source_root/.venv/pyvenv.cfg"
/usr/bin/python3 -I -c 'import sys,venv; assert sys.version_info[:2] == (3,12)'
if [[ -n "$(find "$source_root/agent_runtime" "$source_root/rag" "$source_root/configs" -type l -print -quit)" ]]; then
  echo 'Application source must not contain symlinks to external files.' >&2
  exit 1
fi
install -d -m 0755 /opt/praxis/app
cp -R -- "$source_root/agent_runtime" "$source_root/rag" "$source_root/configs" /opt/praxis/app/
install -m 0644 "$source_root/scripts/ubuntu/server-models.yaml" /opt/praxis/server-models.yaml
for script in launch-server-agent.py provision-model-secrets.py validate-model-venv.py; do
  install -m 0644 "$source_root/scripts/ubuntu/$script" "/opt/praxis/$script"
done
install -m 0644 "$source_root/scripts/ubuntu/requirements-model-service.txt" /opt/praxis/requirements-model-service.txt
# Dedicated OS interpreter, fresh venv, immutable locked wheels; never copy editable Agent hooks here.
/usr/bin/python3 -I -m venv /opt/praxis/model-venv
install -d -m 0700 -o root -g root "$model_cache"
/opt/praxis/model-venv/bin/python -I -m pip --isolated --disable-pip-version-check install \
  --cache-dir "$model_cache" --index-url https://pypi.org/simple --require-hashes --only-binary=:all: \
  -r /opt/praxis/requirements-model-service.txt
/opt/praxis/model-venv/bin/python -I /opt/praxis/validate-model-venv.py
# The Agent's larger environment is separate and never receives provider keys.
cp -a -- "$source_root/.venv" /opt/praxis/agent-venv
/usr/bin/python3 -I /opt/praxis/validate-model-venv.py --relocate-agent /opt/praxis/agent-venv/pyvenv.cfg
for executable in python python3 python3.12; do
  rm -f -- "/opt/praxis/agent-venv/bin/$executable"
  ln -s /usr/bin/python3 "/opt/praxis/agent-venv/bin/$executable"
done
chown -R root:root /opt/praxis
chmod -R go-w /opt/praxis
useradd --system --user-group --home-dir /var/lib/praxis-model --shell /usr/sbin/nologin praxis-model
useradd --system --user-group --create-home --home-dir /var/lib/praxis-agent --shell /bin/bash praxis-agent
chmod 0700 /var/lib/praxis-agent
install -d -m 0700 -o praxis-agent -g praxis-agent /var/lib/praxis-agent/workspace
install -m 0755 "$source_root/scripts/ubuntu/start-server-agent.sh" /usr/local/bin/praxis-agent
install -m 0755 "$source_root/scripts/ubuntu/praxis.sh" /usr/local/bin/praxis
install -m 0644 "$source_root/scripts/ubuntu/praxis-model.service" /etc/systemd/system/praxis-model.service
/usr/bin/python3 -I /opt/praxis/provision-model-secrets.py
systemctl daemon-reload
systemctl enable --now praxis-model.service
systemctl is-active --quiet praxis-model.service
printf '%s\n' 'Service active. Enter terminal chat with: praxis (model menu: /model; help: /help)'
