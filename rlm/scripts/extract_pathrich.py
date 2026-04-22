"""Extract path-rich traces from the full grepfruit JSONL.

The full dataset is a mix of three sources: ~550K StackExchange records with
flat filenames (question.md, answer_N.md), a Wikipedia chunk with section
names as files, and — starting around record 600K — real code repos and web
crawls with deeply nested paths. The flat half makes the dataset feel
homogeneous; the nested half is where the interesting failure modes live.

This script streams the source file once, keeps only records whose documents
contain at least one nested path (``/`` in the filename) past a minimum
depth, and writes them to a new JSONL.

Performance
-----------
Two optimizations over a naive implementation, both preserving output:

1. **Byte-level prefilter.** Before we parse a line, we check whether it
   contains ``"path":"<something>/<something>"``. This is a cheap regex over
   the raw bytes that rules out every flat StackExchange/Wikipedia record
   without paying the JSON parse cost (records can be hundreds of KB).

2. **orjson.** A drop-in faster JSON parser (~3-5x stdlib ``json`` on large
   payloads). Falls back to stdlib if not installed.

On a 316 GB input a naive pass takes ~18 minutes; with these two it drops to
a few minutes, bottlenecked on sequential disk read.

Usage
-----

    uv run python projects/halo/scripts/extract_pathrich.py \
        --src /home/jianbo/catalyst-train/grepfruit.jsonl \
        --dst /home/jianbo/catalyst-train/grepfruit_pathrich.jsonl \
        --target 60000
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

try:
    import orjson as _orjson

    def _loads(data: bytes) -> Any:
        return _orjson.loads(data)

    HAS_ORJSON = True
except ImportError:  # pragma: no cover
    import json as _json

    def _loads(data: bytes) -> Any:
        return _json.loads(data)

    HAS_ORJSON = False


# Match ``"path": "<>/<>"`` with at least one slash inside the quoted value.
# The whole record lands on one JSONL line, so a single regex scan of the
# raw bytes tells us whether there's any nested document path. Using bytes
# avoids the UTF-8 decode.
_PATH_WITH_SLASH = re.compile(rb'"path"\s*:\s*"[^"\\]{3,400}/[^"\\]{0,400}"')


def is_path_rich(record: dict[str, Any], *, min_nested: int = 1, min_max_len: int = 20) -> bool:
    docs = record.get("documents") or []
    nested = 0
    longest = 0
    for d in docs:
        p = d.get("path") if isinstance(d, dict) else None
        if not isinstance(p, str):
            continue
        longest = max(longest, len(p))
        if "/" in p:
            nested += 1
    return nested >= min_nested and longest >= min_max_len


def _prefilter(line: bytes) -> bool:
    """Cheap bytes-only check: could this line possibly be path-rich?"""
    # Fast path 0: if the line has no '/' at all, documents can't be nested.
    if b"/" not in line:
        return False
    # Targeted path-regex on the raw bytes.
    return _PATH_WITH_SLASH.search(line) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--target", type=int, default=60000,
                    help="Stop after this many path-rich records.")
    ap.add_argument("--min-nested", type=int, default=1,
                    help="Require at least this many paths containing '/'.")
    ap.add_argument("--min-max-len", type=int, default=20,
                    help="Require the longest doc path to be at least this long.")
    ap.add_argument("--progress-every", type=int, default=50000,
                    help="Log progress every N scanned lines.")
    args = ap.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[extract] orjson={'yes' if HAS_ORJSON else 'no'}")

    kept = 0
    scanned = 0
    parses = 0
    started = time.time()
    src_size = args.src.stat().st_size if args.src.exists() else 0

    with open(args.src, "rb") as src, open(args.dst, "wb") as dst:
        while True:
            line = src.readline()
            if not line:
                break
            scanned += 1

            if _prefilter(line):
                parses += 1
                try:
                    record = _loads(line)
                except Exception:
                    record = None
                if record is not None and is_path_rich(
                    record,
                    min_nested=args.min_nested,
                    min_max_len=args.min_max_len,
                ):
                    dst.write(line)
                    kept += 1
                    if kept >= args.target:
                        break

            if scanned % args.progress_every == 0:
                elapsed = time.time() - started
                pos = src.tell()
                pct = (pos / src_size * 100) if src_size else 0
                rate = scanned / elapsed if elapsed else 0
                parse_rate = parses / elapsed if elapsed else 0
                print(
                    f"  scanned={scanned:,} parses={parses:,} kept={kept:,} "
                    f"pos={pos / 1e9:,.1f}GB ({pct:.1f}%) "
                    f"rate={rate:,.0f} lines/s parse={parse_rate:,.0f}/s "
                    f"elapsed={elapsed:,.1f}s",
                    flush=True,
                )

    elapsed = time.time() - started
    print(f"Done. scanned={scanned:,} parses={parses:,} kept={kept:,} elapsed={elapsed:,.1f}s")
    print(f"Output: {args.dst}  size={args.dst.stat().st_size / 1e9:.2f}GB")


if __name__ == "__main__":
    main()
