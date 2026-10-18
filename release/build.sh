#!/bin/sh
set -eu
# Container execution is deliberately mandatory for an accepted build.
: "${PINNED_BUILD_IMAGE:?Supply an independently verified Python 3.14.3 image@sha256:digest}"
case "$PINNED_BUILD_IMAGE" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) exit 2;; esac
commit="$1"; wheels="$2"; output="$3"
git rev-parse --verify "$commit^{commit}" >/dev/null
SOURCE_DATE_EPOCH="$(git show -s --format=%ct "$commit")"; export SOURCE_DATE_EPOCH
mkdir -p "$output"
git archive --format=tar "$commit" > "$output/source.tar"
git show "$commit:release/build_payload.py" > "$output/build_payload.py"
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
 -v "$(pwd)/$output:/out" -v "$(pwd)/$wheels:/wheels:ro" \
 "$PINNED_BUILD_IMAGE" \
 python /out/build_payload.py --source /out/source.tar --wheels /wheels --out /out/application.tar.gz --epoch "$SOURCE_DATE_EPOCH"
