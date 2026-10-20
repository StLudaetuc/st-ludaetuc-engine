#!/bin/sh
set -eu
# Sole supported launcher; explicit PORT and immutable runtime contract.
if [ "$#" -ne 0 ]; then echo 'launcher accepts no arguments' >&2; exit 2; fi
: "${PORT:?PORT required}"
case "$PORT" in *[!0-9]*|'') echo 'invalid PORT' >&2; exit 2;; esac
if [ "${#PORT}" -gt 5 ] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then echo 'invalid PORT' >&2; exit 2; fi
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
python -I -B release/runtime_preflight.py
exec python -I -B -m uvicorn app:app --app-dir "$PWD" --host 0.0.0.0 --port "$PORT"
