#!/bin/sh
set -eu

APP_NAME="HALO"
CHANNEL="${HALO_CHANNEL:-stable}"
RELEASE_BASE_URL="${HALO_RELEASE_BASE_URL:-https://inference.net/halo/releases}"
REQUIRE_SIGNATURE="${HALO_REQUIRE_SIGNATURE:-0}"
NO_OPEN="${HALO_NO_OPEN:-0}"
VERBOSE="${HALO_VERBOSE:-0}"
SKIP_ENGINE_INSTALL="${HALO_SKIP_ENGINE_INSTALL:-0}"
HALO_ENGINE_REPO_URL="${HALO_ENGINE_REPO_URL:-https://github.com/context-labs/HALO}"

if [ "$VERBOSE" = "1" ]; then
  set -x
fi

main() {
  case "$CHANNEL" in
    stable|canary) ;;
    *) fail "Unsupported HALO_CHANNEL: $CHANNEL. Use stable or canary." ;;
  esac

  require_command uname
  require_downloader
  require_command tar
  require_command find

  os="$(uname -s)"
  machine="$(uname -m)"
  app_file_name="$APP_NAME"
  if [ "$CHANNEL" != "stable" ]; then
    app_file_name="$APP_NAME-$CHANNEL"
  fi

  case "$os:$machine" in
    Darwin:arm64)
      platform="macos"
      arch="arm64"
      artifact="$CHANNEL-$platform-$arch-$app_file_name.dmg"
      install_macos=1
      ;;
    Linux:x86_64|Linux:amd64)
      assert_debian_like_linux
      platform="linux"
      arch="x64"
      artifact="$CHANNEL-$platform-$arch-$app_file_name-Setup.tar.gz"
      install_macos=0
      ;;
    Darwin:*)
      fail "HALO v1 supports Apple Silicon Macs only. Detected macOS architecture: $machine."
      ;;
    Linux:*)
      fail "HALO v1 supports Ubuntu/Debian x64 only. Detected Linux architecture: $machine."
      ;;
    *)
      fail "Unsupported operating system: $os."
      ;;
  esac

  release_url="$(trim_slashes "$RELEASE_BASE_URL")/$CHANNEL"
  tmp_dir="$(mktemp -d 2>/dev/null || mktemp -d -t halo-install)"
  trap 'cleanup "$tmp_dir"' EXIT INT TERM

  artifact_path="$tmp_dir/$artifact"
  checksums_path="$tmp_dir/SHA256SUMS"
  signature_bundle_path="$tmp_dir/SHA256SUMS.sigstore.json"

  log "Downloading $artifact"
  download "$release_url/$artifact" "$artifact_path"
  download "$release_url/SHA256SUMS" "$checksums_path"

  if download_optional "$release_url/SHA256SUMS.sigstore.json" "$signature_bundle_path"; then
    verify_signature "$checksums_path" "$signature_bundle_path"
  elif [ "$REQUIRE_SIGNATURE" = "1" ]; then
    fail "Signature bundle is required but was not found at $release_url/SHA256SUMS.sigstore.json"
  else
    warn "Signature bundle not found; continuing with SHA256 verification."
  fi

  verify_checksum "$checksums_path" "$artifact" "$artifact_path"

  if [ "$install_macos" = "1" ]; then
    install_on_macos "$artifact_path"
  else
    install_on_linux "$artifact_path"
  fi
}

install_on_macos() {
  dmg_path="$1"
  require_command hdiutil
  require_command find

  mount_dir="$(mktemp -d 2>/dev/null || mktemp -d -t halo-dmg)"
  log "Mounting $dmg_path"
  hdiutil attach "$dmg_path" -mountpoint "$mount_dir" -nobrowse -quiet

  app_path="$(find "$mount_dir" -maxdepth 1 -type d -name "$APP_NAME*.app" | head -n 1)"
  if [ -z "$app_path" ]; then
    hdiutil detach "$mount_dir" -quiet || true
    fail "Mounted DMG did not contain a HALO app bundle."
  fi

  app_bundle_name="$(basename "$app_path")"
  install_dir="${HALO_INSTALL_DIR:-/Applications}"
  if [ ! -w "$install_dir" ]; then
    install_dir="$HOME/Applications"
  fi
  mkdir -p "$install_dir"

  dest="$install_dir/$app_bundle_name"
  log "Installing $app_bundle_name to $install_dir"
  preserve_legacy_macos_data "$dest"
  rm -rf "$dest"
  cp -R "$app_path" "$dest"
  hdiutil detach "$mount_dir" -quiet

  log "Installed $APP_NAME at $dest"
  bootstrap_halo_engine macos
  if [ "$NO_OPEN" != "1" ]; then
    open "$dest" >/dev/null 2>&1 || true
  fi
}

preserve_legacy_macos_data() {
  existing_app="$1"
  legacy_data_dir="$existing_app/Contents/MacOS/data"
  support_dir="${HALO_APP_DATA_DIR:-$HOME/Library/Application Support/net.inference.halo}"
  db_name="halo-canvas.sqlite"

  if [ ! -d "$legacy_data_dir" ]; then
    return
  fi

  mkdir -p "$support_dir"
  chmod 700 "$support_dir" >/dev/null 2>&1 || true

  if [ -f "$legacy_data_dir/$db_name" ] && [ ! -s "$support_dir/$db_name" ]; then
    log "Preserving existing HALO data in $support_dir"
    for file in "$legacy_data_dir"/"$db_name"*; do
      [ -f "$file" ] || continue
      cp -p "$file" "$support_dir/$(basename "$file")"
    done
  fi

  for dir_name in halo-engine halo-runs; do
    if [ -d "$legacy_data_dir/$dir_name" ] && [ ! -e "$support_dir/$dir_name" ]; then
      cp -R "$legacy_data_dir/$dir_name" "$support_dir/$dir_name"
    fi
  done
}

install_on_linux() {
  archive_path="$1"
  extract_dir="$(mktemp -d 2>/dev/null || mktemp -d -t halo-linux)"
  tar -xzf "$archive_path" -C "$extract_dir"

  installer="$extract_dir/installer"
  if [ ! -x "$installer" ]; then
    fail "Linux archive did not contain an executable installer."
  fi

  if [ -n "${HALO_INSTALL_DIR:-}" ]; then
    mkdir -p "$HALO_INSTALL_DIR"
    export XDG_DATA_HOME="$HALO_INSTALL_DIR"
  fi

  log "Running HALO Linux installer"
  "$installer"
  bootstrap_halo_engine linux
  log "Installed $APP_NAME"
}

bootstrap_halo_engine() {
  platform="$1"
  if [ "$SKIP_ENGINE_INSTALL" = "1" ]; then
    log "Skipping HALO engine bootstrap because HALO_SKIP_ENGINE_INSTALL=1"
    return
  fi

  if ! command -v git >/dev/null 2>&1; then
    warn "git not found; HALO can install the engine later from Settings or first analysis run."
    return
  fi
  if ! command -v uv >/dev/null 2>&1; then
    warn "uv not found; HALO can install the engine later after uv is available."
    return
  fi

  case "$platform" in
    macos)
      app_data_dir="${HALO_APP_DATA_DIR:-$HOME/Library/Application Support/net.inference.halo}"
      ;;
    *)
      app_data_dir="${HALO_APP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/net.inference.halo}"
      ;;
  esac

  engine_dir="${HALO_ENGINE_DIR:-$app_data_dir/halo-engine}"
  mkdir -p "$app_data_dir"

  if [ -d "$engine_dir/.git" ]; then
    log "Updating HALO engine in $engine_dir"
    if ! git -C "$engine_dir" pull --ff-only; then
      warn "Could not update existing HALO engine; the app can retry from Settings."
      return
    fi
  elif [ ! -e "$engine_dir" ] || is_empty_dir "$engine_dir"; then
    rm -rf "$engine_dir"
    log "Downloading HALO engine to $engine_dir"
    if ! git clone "$HALO_ENGINE_REPO_URL" "$engine_dir"; then
      warn "Could not download HALO engine; the app can retry from Settings."
      return
    fi
  else
    warn "$engine_dir exists but is not a git checkout; skipping engine bootstrap."
    return
  fi

  log "Installing HALO engine dependencies with uv"
  if ! (cd "$engine_dir" && uv sync); then
    warn "Could not install HALO engine dependencies; the app can retry from Settings."
    return
  fi
}

is_empty_dir() {
  dir="$1"
  [ -d "$dir" ] || return 1
  [ -z "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit)" ]
}

verify_signature() {
  checksums_path="$1"
  signature_bundle_path="$2"
  if ! command -v cosign >/dev/null 2>&1; then
    if [ "$REQUIRE_SIGNATURE" = "1" ]; then
      fail "cosign is required for signature verification. Install cosign or unset HALO_REQUIRE_SIGNATURE."
    fi
    warn "cosign not found; continuing with SHA256 verification."
    return
  fi

  identity_regexp="${HALO_COSIGN_IDENTITY_REGEXP:-https://github.com/context-labs/HALO/.*}"
  log "Verifying SHA256SUMS signature with cosign"
  cosign verify-blob "$checksums_path" \
    --bundle "$signature_bundle_path" \
    --certificate-identity-regexp "$identity_regexp" \
    --certificate-oidc-issuer "https://token.actions.githubusercontent.com" >/dev/null
}

verify_checksum() {
  checksums_path="$1"
  artifact="$2"
  artifact_path="$3"
  expected="$(awk -v artifact="$artifact" '$2 == artifact { print $1 }' "$checksums_path" | head -n 1)"
  if [ -z "$expected" ]; then
    fail "No checksum found for $artifact"
  fi
  actual="$(sha256_file "$artifact_path")"
  if [ "$expected" != "$actual" ]; then
    fail "Checksum mismatch for $artifact"
  fi
  log "Verified SHA256 checksum"
}

sha256_file() {
  file_path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file_path" | awk '{ print $1 }'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file_path" | awk '{ print $1 }'
  else
    fail "sha256sum or shasum is required."
  fi
}

assert_debian_like_linux() {
  if [ ! -r /etc/os-release ]; then
    fail "HALO v1 Linux install supports Ubuntu/Debian-like systems with /etc/os-release."
  fi
  os_release="$(cat /etc/os-release)"
  case "$os_release" in
    *"ID=ubuntu"*|*"ID=debian"*|*"ID_LIKE=debian"*|*"ID_LIKE=\"debian\""*) ;;
    *) fail "HALO v1 Linux install supports Ubuntu/Debian-like systems only." ;;
  esac
}

require_downloader() {
  if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
  elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
  else
    fail "curl or wget is required."
  fi
}

download() {
  url="$1"
  output="$2"
  if [ "$DOWNLOADER" = "curl" ]; then
    curl -fsSL "$url" -o "$output"
  else
    wget -q "$url" -O "$output"
  fi
}

download_optional() {
  url="$1"
  output="$2"
  if [ "$DOWNLOADER" = "curl" ]; then
    curl -fsSL "$url" -o "$output" >/dev/null 2>&1
  else
    wget -q "$url" -O "$output" >/dev/null 2>&1
  fi
}

trim_slashes() {
  printf "%s" "$1" | sed 's:/*$::'
}

cleanup() {
  tmp_dir="$1"
  rm -rf "$tmp_dir"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "$1 is required."
  fi
}

log() {
  printf '%s\n' "halo-install: $*"
}

warn() {
  printf '%s\n' "halo-install: warning: $*" >&2
}

fail() {
  printf '%s\n' "halo-install: error: $*" >&2
  exit 1
}

main "$@"
