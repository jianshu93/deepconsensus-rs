#!/usr/bin/env bash
set -euo pipefail
root="${1:-fixtures}"
mkdir -p "$root"
if [[ ! -d "$root/upstream/.git" ]]; then
  git clone --depth 1 --branch r1.2 https://github.com/google/deepconsensus "$root/upstream"
fi
commit="$(git -C "$root/upstream" rev-parse HEAD)"
if [[ "$commit" != 979108ddbc0a950ea493071690507ac32c8ee9cb ]]; then
  printf 'Unexpected upstream fixture revision: %s\n' "$commit" >&2
  exit 1
fi
cp -R "$root/upstream/deepconsensus/testdata/human_1m" "$root/"
printf 'Fixtures: %s/human_1m\n' "$root"
