#!/bin/sh
set -eu
exec python -m uvicorn app:app --host 127.0.0.1 --port "${PORT:-8000}"
