"""IR-05 fail-fast runtime/lock guard, stdlib only; no engine or database import.

Installed RECORD verification detects drift, not hostile rewriting of both files and
RECORD. Immutable image provenance and a trusted verifier remain separate gates.
"""
import argparse
import base64
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def norm(name):
    return re.sub(r'[-_.]+', '-', name).lower()

def verify(root=ROOT, phase='runtime'):
    c = json.loads((root / 'release/runtime_contract.json').read_text())
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    check(platform.python_version() == c['python'], 'requires exact Python ' + c['python'])
    check(platform.python_implementation() == c['implementation'], 'requires CPython')
    check(platform.system() == c['system'] and platform.machine() == c['machine'], 'requires Linux x86_64')
    check('PYTHON_VERSION' not in os.environ or os.environ['PYTHON_VERSION'] == c['python'], 'PYTHON_VERSION must be absent or exact approved version')
    lock = (root / 'requirements.txt').read_bytes()
    check(hashlib.sha256(lock).hexdigest() == c['lock_sha256'], 'requirements lock digest mismatch')
    check(lock == (root / 'release/G.lock').read_bytes(), 'requirements must equal selected G lock')
    expected = {norm(k): v for k, v in c['distributions'].items()}
    parsed = {}
    for line in lock.decode().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:[0-9a-f]{64}', line)
        check(match is not None, 'invalid hash-locked requirement')
        if match:
            parsed[norm(match[1])] = match[2]
    check(parsed == expected, 'lock distribution inventory mismatch')
    if phase == 'runtime':
        installed = {}
        for d in md.distributions():
            name = norm(d.metadata['Name'])
            check(name not in installed, 'duplicate installed distribution: ' + name)
            installed[name] = d
        check(set(installed) <= set(expected) | {'pip', 'setuptools'}, 'unexpected runtime distributions: ' + ','.join(sorted(set(installed) - set(expected) - {'pip', 'setuptools'})))
        for name, version in expected.items():
            d = installed.get(name)
            if d is None:
                errors.append('missing distribution: ' + name)
                continue
            check(d.version == version, 'version mismatch: ' + name)
            files = d.files
            check(bool(files), 'missing RECORD inventory: ' + name)
            for f in files or []:
                if f.hash is None:
                    check(str(f).endswith('/RECORD') or str(f).endswith('.pyc'), 'unhashed installed file: ' + str(f))
                    continue
                try:
                    content = d.locate_file(f).read_bytes()
                    got = base64.urlsafe_b64encode(hashlib.new(f.hash.mode, content).digest()).rstrip(b'=').decode()
                    check(got == f.hash.value, 'installed file digest mismatch: ' + name + '/' + str(f))
                except (OSError, ValueError):
                    errors.append('installed file unreadable: ' + name + '/' + str(f))
    return {'phase': phase, 'python': platform.python_version(), 'machine': platform.machine(), 'errors': errors, 'status': 'FAIL' if errors else 'PASS'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['build', 'runtime'], default='runtime')
    a = p.parse_args()
    result = verify(phase=a.phase)
    print(json.dumps(result, sort_keys=True))
    sys.exit(bool(result['errors']))
