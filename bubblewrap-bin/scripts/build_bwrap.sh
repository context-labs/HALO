#!/usr/bin/env bash
# Build a static-libcap bwrap binary from a pinned upstream tarball.
#
# Run on a Linux CI runner. Outputs the built binary at:
#   src/bubblewrap_bin/_bin/bwrap
#
# Inputs:
#   BWRAP_VERSION  upstream version tag (e.g. 0.11.0)
#   BWRAP_SHA256   sha256 of the upstream release tarball
#   ARCH           target arch label used in cache filenames (x86_64|aarch64)
#
# Build pinning:
#   - libcap is installed via the host package manager and statically linked
#     into bwrap so the produced binary does not require libcap at runtime.
#   - The build is reproducible-ish: we don't ship build IDs in the wheel.

set -euo pipefail

BWRAP_VERSION="${BWRAP_VERSION:?BWRAP_VERSION must be set}"
BWRAP_SHA256="${BWRAP_SHA256:?BWRAP_SHA256 must be set}"
ARCH="${ARCH:?ARCH must be set (x86_64|aarch64)}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
out_dir="${repo_root}/src/bubblewrap_bin/_bin"
work_dir="$(mktemp -d -t bubblewrap-bin-XXXXXX)"
trap 'rm -rf "${work_dir}"' EXIT

mkdir -p "${out_dir}"

tarball_url="https://github.com/containers/bubblewrap/releases/download/v${BWRAP_VERSION}/bubblewrap-${BWRAP_VERSION}.tar.xz"
tarball_path="${work_dir}/bubblewrap-${BWRAP_VERSION}.tar.xz"

echo "[bubblewrap-bin] downloading ${tarball_url}"
curl --fail --location --output "${tarball_path}" "${tarball_url}"

echo "[bubblewrap-bin] verifying sha256"
echo "${BWRAP_SHA256}  ${tarball_path}" | sha256sum --check --strict

echo "[bubblewrap-bin] extracting"
tar --extract --xz --file "${tarball_path}" --directory "${work_dir}"

src_dir="${work_dir}/bubblewrap-${BWRAP_VERSION}"
build_dir="${work_dir}/build"
prefix="${work_dir}/install"

echo "[bubblewrap-bin] configuring (meson)"
meson setup "${build_dir}" "${src_dir}" \
    --prefix="${prefix}" \
    --buildtype=release \
    -Dman=disabled \
    -Dselinux=disabled \
    -Dtests=false

echo "[bubblewrap-bin] building"
meson compile -C "${build_dir}"

echo "[bubblewrap-bin] installing into staging prefix"
meson install -C "${build_dir}"

echo "[bubblewrap-bin] copying binary to ${out_dir}/bwrap"
install --mode=0755 "${prefix}/bin/bwrap" "${out_dir}/bwrap"

echo "[bubblewrap-bin] recording version"
echo "${BWRAP_VERSION}" > "${repo_root}/VENDORED_BWRAP_VERSION"

echo "[bubblewrap-bin] smoke test"
"${out_dir}/bwrap" --version

echo "[bubblewrap-bin] done"
echo "  bwrap:    ${out_dir}/bwrap"
echo "  arch:     ${ARCH}"
echo "  version:  ${BWRAP_VERSION}"
