#!/usr/bin/env zsh
# Builds a standalone python 3.12 venv at engine/.sandbox-venv containing
# halo-engine + numpy + pandas. Idempotent: re-runs are a no-op if present.

set -euo pipefail

here="${0:a:h}"
root="${here}/.."
venv="${root}/.sandbox-venv"

if [[ -d "${venv}/bin" ]]; then
  echo "sandbox venv already exists at ${venv}"
  exit 0
fi

uv venv --python 3.12 "${venv}"
"${venv}/bin/pip" install --no-cache-dir \
  "numpy>=2.0" \
  "pandas>=2.2" \
  "pydantic>=2.8"
"${venv}/bin/pip" install --no-cache-dir -e "${root}"

echo "sandbox venv built at ${venv}"
