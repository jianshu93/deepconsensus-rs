#!/usr/bin/env bash
set -euo pipefail
root="${1:-models}"
original="$root/original-v1.2"
mkdir -p "$original"
base="https://storage.googleapis.com/brain-genomics-public/research/deepconsensus/models/v1.2/model_checkpoint"
for name in checkpoint.index checkpoint.data-00000-of-00001 params.json; do
  curl --fail --location --retry 3 "$base/$name" --output "$original/$name.tmp"
  mv "$original/$name.tmp" "$original/$name"
done
(
  cd "$original"
  if command -v sha256sum >/dev/null; then
    sha_cmd=(sha256sum)
  else
    sha_cmd=(shasum -a 256)
  fi
  printf '%s\n' \
    '42f42f87c0df18386c0c8d1906d7eacd115fe1ee5507cd27275dcb594ba67c11  checkpoint.index' \
    '49a122bc456eea2f402c9e2d52a456012c0384a8e7a6d95902568c56e0fd8cc8  checkpoint.data-00000-of-00001' \
    'c1b022a73e16c3d8cb876192c427a4c3ad1b7b7d0370b92fb430097068e1e7bc  params.json' | "${sha_cmd[@]}" --check
)
args=(import-tf --checkpoint "$original/checkpoint" --params "$original/params.json" --output "$root/v1.2")
if [[ -x target/release/deepconsensus-model ]]; then
  target/release/deepconsensus-model "${args[@]}"
else
  cargo +stable run --release --locked --bin deepconsensus-model -- "${args[@]}"
fi
