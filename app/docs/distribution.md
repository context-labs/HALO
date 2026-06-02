# HALO Distribution

HALO v1 is distributed as a signed macOS Apple Silicon desktop app and a Linux x64 desktop app for Ubuntu/Debian. Users install it with:

```bash
curl -fsSL https://inference.net/halo/install.sh | sh
```

## Release Targets

- App display name: `HALO`
- Bundle identifier: `net.inference.halo`
- Installer URL: `https://inference.net/halo/install.sh`
- Release root: `https://inference.net/halo/releases`
- Stable artifacts: `https://inference.net/halo/releases/stable/`
- Canary artifacts: `https://inference.net/halo/releases/canary/`
- v1 platforms:
  - `macos-arm64`
  - `linux-x64`

Windows and macOS Intel builds are intentionally out of scope for v1.

## Local Build Commands

Development bundle:

```bash
bun run build:desktop
```

Unsigned local stable dry run:

```bash
HALO_SKIP_CODESIGN=1 bun run build:stable
bun run release:manifest -- --channel stable --artifacts-dir artifacts
bun run release:verify -- --channel stable --artifacts-dir artifacts
```

Production stable build on CI:

```bash
bun run build:stable
bun run release:manifest -- --channel stable --artifacts-dir artifacts
bun run release:verify -- --channel stable --artifacts-dir artifacts
```

CI macOS release builds intentionally fail if signing or notarization secrets are
missing. Use `HALO_SKIP_CODESIGN=1` only for local dry runs, never for a
published macOS release.

Production canary build:

```bash
bun run build:canary
bun run release:manifest -- --channel canary --artifacts-dir artifacts
bun run release:verify -- --channel canary --artifacts-dir artifacts
```

## Required Release Secrets

macOS signing and notarization:

| Secret | Purpose |
| --- | --- |
| `ELECTROBUN_DEVELOPER_ID` | Developer ID Application signing identity name. |
| `APPLE_DEVELOPER_ID_CERTIFICATE_BASE64` | Base64-encoded `.p12` Developer ID certificate. |
| `APPLE_DEVELOPER_ID_CERTIFICATE_PASSWORD` | Password for the `.p12` certificate. |
| `APPLE_KEYCHAIN_PASSWORD` | Temporary CI keychain password. |
| `ELECTROBUN_APPLEAPIISSUER` | App Store Connect API issuer UUID. |
| `ELECTROBUN_APPLEAPIKEY` | App Store Connect API key ID. |
| `APPLE_API_KEY_P8_BASE64` | Base64-encoded App Store Connect `.p8` key. |
| `ELECTROBUN_APPLEID` | Optional fallback Apple ID for notarization. |
| `ELECTROBUN_APPLEIDPASS` | Optional fallback app-specific password for notarization. |
| `ELECTROBUN_TEAMID` | Optional fallback Apple Developer Team ID for notarization. |

Prefer the App Store Connect API key secrets for notarization. The Apple ID
fallback is supported because ElectroBun also supports `notarytool` with an
Apple ID, app-specific password, and team ID.

Before triggering a release, verify the Apple team can notarize:

```bash
xcrun notarytool history \
  --apple-id "$ELECTROBUN_APPLEID" \
  --password "$ELECTROBUN_APPLEIDPASS" \
  --team-id "$ELECTROBUN_TEAMID"
```

If this reports a missing or expired agreement, sign in to Apple Developer or
App Store Connect as the account holder/admin and accept the pending legal
agreement before rerunning CI.

Static hosting publish:

| Secret | Purpose |
| --- | --- |
| `HALO_RELEASE_BUCKET` | Bucket name for release artifacts. |
| `HALO_RELEASE_ENDPOINT_URL` | S3-compatible endpoint URL. |
| `HALO_RELEASE_ACCESS_KEY_ID` | Upload access key. |
| `HALO_RELEASE_SECRET_ACCESS_KEY` | Upload secret key. |

The workflow signs `SHA256SUMS` with keyless Sigstore/cosign using GitHub OIDC.

## Artifact Layout

Expected stable artifact names:

```text
stable-macos-arm64-HALO.dmg
stable-macos-arm64-HALO.app.tar.zst
stable-macos-arm64-update.json
stable-linux-x64-HALO-Setup.tar.gz
stable-linux-x64-HALO.tar.zst
stable-linux-x64-update.json
manifest.json
SHA256SUMS
SHA256SUMS.sigstore.json
```

Canary uses the same shape with `canary` prefixes and `HALO-canary` app file names.

`manifest.json` includes version, channel, generated timestamp, release base URL, file names, artifact URLs, sizes, checksums, platform, architecture, and artifact kind.

## Release Hosting

GitHub Actions publishes release files into the `halo-releases` R2 bucket under:

```text
halo/releases/<channel>/
halo/install.sh
```

The public `https://inference.net/halo/...` paths are served by the Cloudflare Worker in `app/workers/halo-release-worker.ts`, configured by `app/wrangler.release.toml`.

Deploy the route/bucket proxy after creating or changing the worker:

```bash
cd app
bunx wrangler deploy --config wrangler.release.toml
```

## Installer Behavior

The install script lives at `app/scripts/install.sh` and should be served as `https://inference.net/halo/install.sh`.

Supported environment flags:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HALO_CHANNEL` | `stable` | Install `stable` or `canary`. |
| `HALO_INSTALL_DIR` | `/Applications` on macOS, ElectroBun/XDG default on Linux | Override install location. |
| `HALO_NO_OPEN` | `0` | Set to `1` to avoid opening the app after install. |
| `HALO_VERBOSE` | `0` | Set to `1` for shell tracing. |
| `HALO_REQUIRE_SIGNATURE` | `0` | Set to `1` to require cosign signature verification. |

macOS install downloads the DMG, verifies SHA256, mounts it, and copies the app into `/Applications` or `~/Applications`.

Linux install downloads the ElectroBun installer archive, verifies SHA256, extracts it, and runs the bundled installer. `HALO_INSTALL_DIR` is passed through `XDG_DATA_HOME` because the ElectroBun Linux installer installs into user-local XDG data paths.

## Verification

Before publishing:

```bash
bun run typecheck
bun test
bun run build:web
```

macOS artifact verification:

```bash
codesign --verify --deep --strict --verbose=2 "HALO.app"
codesign -dv --verbose=4 "HALO.app"
spctl --assess --type execute --verbose "HALO.app"
xcrun stapler validate "HALO.app"
codesign --verify --verbose=2 "stable-macos-arm64-HALO.dmg"
spctl --assess --type open --context context:primary-signature --verbose=4 "stable-macos-arm64-HALO.dmg"
xcrun stapler validate "stable-macos-arm64-HALO.dmg"
```

Linux verification:

```bash
HALO_RELEASE_BASE_URL=https://inference.net/halo/releases \
HALO_NO_OPEN=1 \
curl -fsSL https://inference.net/halo/install.sh | sh
```

Then launch HALO and send test spans:

```bash
CATALYST_OTLP_ENDPOINT=http://127.0.0.1:8799/v1/traces bun run fire:test-spans
```

## Rollback

To roll back a broken release, restore the previous channel directory contents on the release host:

```text
https://inference.net/halo/releases/stable/
```

The install script always downloads the current artifact names and verifies against the current `SHA256SUMS`, so rollback is a static-file replacement. Preserve `manifest.json`, `SHA256SUMS`, `SHA256SUMS.sigstore.json`, and ElectroBun `*-update.json` files together.
