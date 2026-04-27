#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ENGINE_ROOT="$REPO_ROOT/engine"
GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir)"
HOOKS_DIR="$GIT_DIR/hooks"

mkdir -p "$HOOKS_DIR"

install_hook() {
  local hook_name="$1"
  local source_path="$ENGINE_ROOT/scripts/git-hooks/$hook_name"
  local target_path="$HOOKS_DIR/$hook_name"

  cp "$source_path" "$target_path"
  chmod +x "$target_path"
  echo "[hooks] Installed $hook_name"
}

install_hook pre-commit
install_hook pre-push

echo "[hooks] Engine git hooks installed in $HOOKS_DIR"
