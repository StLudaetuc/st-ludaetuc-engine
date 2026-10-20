#!/bin/sh
set -eu
# Local build only. No push, deployment, credentials or registry mutation.
if [ "$#" -ne 4 ]; then echo 'usage: build_image.sh FULL_SOURCE_SHA WHEELHOUSE NEW_CONTEXT_DIR NEW_OCI_TAR' >&2; exit 2; fi
sha=$1; wheels=$2; context=$3; output=$4
case "$sha" in *[!0-9a-f]*|'') exit 2;; esac
[ "${#sha}" -eq 40 ]
[ ! -e "$context" ] && [ ! -e "$output" ]
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
[ "$(git -C "$root" rev-parse "$sha^{commit}")" = "$sha" ]
mkdir "$context"; mkdir "$context/source" "$context/wheelhouse"
git -C "$root" archive "$sha" | tar -xf - -C "$context/source"
python - "$context/source/release/RELEASE_MANIFEST.json" "$wheels" "$context/wheelhouse" <<'PY'
import hashlib,json,pathlib,shutil,sys
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
for w in m['wheels']:
 p=pathlib.Path(sys.argv[2])/w['wheel'];assert hashlib.sha256(p.read_bytes()).hexdigest()==w['sha256'];shutil.copyfile(p,pathlib.Path(sys.argv[3])/w['wheel'])
PY
epoch=$(git -C "$root" show -s --format=%ct "$sha")
docker buildx build --platform linux/amd64 --network=none --provenance=false --sbom=false --build-arg "SOURCE_DATE_EPOCH=$epoch" --output "type=oci,dest=$output,rewrite-timestamp=true" -f "$context/source/release/Dockerfile.rc2" "$context"
# OCI metadata must be independently inspected/attested. A successful tar export is
# not a tested deployment. Repeat with a second NEW context/output and compare digests.
