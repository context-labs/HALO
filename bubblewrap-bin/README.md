# bubblewrap-bin

A Python wheel that ships a prebuilt
[bubblewrap (`bwrap`)](https://github.com/containers/bubblewrap) executable
for Linux. Useful for projects that need an OS sandbox at runtime without
asking users to `apt install bubblewrap` first.

## Install

```sh
pip install bubblewrap-bin
```

Prebuilt wheels are published for Linux on `x86_64` and `aarch64`. Other
platforms can install the sdist but it does not include the binary.

## Usage

```python
import subprocess
from bubblewrap_bin import bwrap_path

subprocess.run([str(bwrap_path()), "--version"], check=True)
```

`bwrap_path()` returns the absolute path of the bundled `bwrap` executable.
It raises `BubblewrapNotBundledError` when no binary is packaged in the
installed distribution (typically when installed from sdist on an
unsupported platform).

## Sandbox limitations

`bwrap` from this package still requires kernel/runtime support for
unprivileged user namespaces. On hosts that disable or restrict that
support (some hardened distros, locked-down Docker / Kubernetes
environments, Ubuntu 24.04+ AppArmor `unprivileged_userns_clone`
restrictions) the binary exists but the sandbox will fail to set up.

That's a host-level policy decision and cannot be fixed by repackaging.
The simplest workaround on those hosts is to install bubblewrap from the
distro package manager — `sudo apt install bubblewrap` (or `dnf` / `apk`
equivalents) — which ships with the policy hooks the distro maintainer
configured. The packaged binary in this wheel is unmodified upstream
bubblewrap and is intended for hosts where unprivileged user namespaces
work out of the box.

## Versioning

The Python package version tracks the upstream `bwrap` version with a
package revision suffix (for example `0.11.0.post1`). The exact upstream
version baked into a given wheel is recorded in `VENDORED_BWRAP_VERSION` at
the repository root.

## License

The Python wrapper is released under LGPL-2.0-or-later to match the
upstream `bubblewrap` project. See `LICENSE` for the full text and
attribution.
