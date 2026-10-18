#!/bin/sh
set -eu
# Only supported recovery launcher: no command/entrypoint arguments accepted.
if [ "$#" -ne 0 ]; then exit 2; fi
exec python -m uvicorn release.recovery_entry:app --host 127.0.0.1 --port "${PORT:-8000}"
