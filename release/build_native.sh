#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
python -I -B release/runtime_preflight.py --phase build
python -m pip install --require-hashes -r requirements.txt
python -I -B release/runtime_preflight.py
