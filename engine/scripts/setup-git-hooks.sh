#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ENGINE_ROOT="$REPO_ROOT/engine"
HOOKS_PATH=".githooks"

git -C "$REPO_ROOT" config core.hooksPath "$HOOKS_PATH"

chmod +x "$REPO_ROOT/.githooks/pre-commit"
chmod +x "$REPO_ROOT/.githooks/pre-push"
chmod +x "$ENGINE_ROOT/scripts/git-hooks/pre-commit"
chmod +x "$ENGINE_ROOT/scripts/git-hooks/pre-push"

echo "[hooks] Configured git hooks path at $HOOKS_PATH"
