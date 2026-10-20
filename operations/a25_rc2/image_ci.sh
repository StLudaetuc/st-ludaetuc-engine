#!/bin/sh
set -eu
# Disposable runner-only builds: no registry push, Render or Production calls.
sha=$(git rev-parse HEAD)
mkdir -p /tmp/rc2-image-evidence
python -m pip download --require-hashes --only-binary=:all: -r requirements.txt -d /tmp/rc2-image-wheels
# Resolve the public builder tool once and pin its observed digest for both builds.
docker pull moby/buildkit:buildx-stable-1
builder_image=$(docker image inspect moby/buildkit:buildx-stable-1 --format '{{index .RepoDigests 0}}')
printf '%s\n' "$builder_image" > /tmp/rc2-image-evidence/builder-image.txt
docker version > /tmp/rc2-image-evidence/docker-version.txt
docker buildx version > /tmp/rc2-image-evidence/buildx-version.txt
trap 'docker buildx rm rc2-builder-one rc2-builder-two >/dev/null 2>&1 || true' EXIT
for round in one two; do
 docker buildx create --name "rc2-builder-$round" --driver docker-container --driver-opt "image=$builder_image" --use
 sh release/build_image.sh "$sha" /tmp/rc2-image-wheels "/tmp/rc2-image-context-$round" "/tmp/rc2-image-evidence/image-$round.oci.tar"
 docker buildx rm "rc2-builder-$round"
done
python operations/a25_rc2/verify_oci.py /tmp/rc2-image-evidence/image-one.oci.tar /tmp/rc2-image-evidence/image-two.oci.tar /tmp/rc2-image-evidence/OCI_ATTESTATION.json
# Both source trees are built independently by the CI matrix, with explicit PORT.
# Exports retained for reviewer load/start/fault drills; no remote image is deployed.
python -m pip freeze > /tmp/rc2-image-evidence/build-client-freeze.txt
python - <<'PY'
import hashlib,json,pathlib,subprocess
r=pathlib.Path('/tmp/rc2-image-evidence');files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.iterdir() if p.is_file()};m={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),'files':files,'base_image':'python:3.14.3-slim@sha256:5e59aae31ff0e87511226be8e2b94d78c58f05216efda3b07dbbed938ec8583b','production':'HOLD','runtime_activation':'NOT_PERFORMED'};(r/'BUILD_RECEIPT.json').write_text(json.dumps(m,indent=2)+'\n')
PY
