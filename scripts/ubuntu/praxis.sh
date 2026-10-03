#!/usr/bin/env bash
# Public terminal entrypoint; provider credentials never enter this process.
set -euo pipefail
set +x
if [[ $# == 0 ]]; then
  set -- chat
fi
case "$1" in
  chat|run)
    if ! systemctl is-active --quiet praxis-model.service; then
      echo 'Model service is inactive. Administrator: sudo systemctl start praxis-model' >&2
      exit 1
    fi
    ;;
esac
terminal="${TERM:-dumb}"
if [[ -t 0 && -t 1 && "$terminal" == dumb ]]; then
  terminal=xterm-256color
fi
if [[ "$(id -un)" == praxis-agent ]]; then
  exec /usr/bin/env -i PATH=/usr/bin:/bin TERM="$terminal" /usr/local/bin/praxis-agent "$@"
fi
exec sudo -H -u praxis-agent -- /usr/bin/env -i PATH=/usr/bin:/bin TERM="$terminal" \
  /usr/local/bin/praxis-agent "$@"
