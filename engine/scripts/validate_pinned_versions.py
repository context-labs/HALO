#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
EXACT_VERSION_RE = re.compile(r"^==\S+$")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def extract_name_and_version(dependency: str) -> tuple[str, str | None]:
    dependency = dependency.split(";")[0].strip()
    dependency = re.sub(r"\[.*?\]", "", dependency)
    match = re.match(r"^([A-Za-z0-9._-]+)\s*(.*)", dependency)
    if not match:
        return dependency, None
    name = match.group(1).strip()
    version = match.group(2).strip() or None
    return name, version


def dependency_strings(section: Any) -> list[str]:
    if not isinstance(section, list):
        return []
    return [item for item in section if isinstance(item, str)]


def main() -> int:
    with PYPROJECT.open("rb") as file:
        data = tomllib.load(file)

    source_pinned = {normalize(name) for name in data.get("tool", {}).get("uv", {}).get("sources", {}).keys()}
    dependencies: list[tuple[str, str, str | None]] = []

    for raw in dependency_strings(data.get("project", {}).get("dependencies", [])):
        name, version = extract_name_and_version(raw)
        dependencies.append(("project.dependencies", name, version))

    optional = data.get("project", {}).get("optional-dependencies", {})
    for section, raw_dependencies in optional.items():
        for raw in dependency_strings(raw_dependencies):
            name, version = extract_name_and_version(raw)
            dependencies.append((f"project.optional-dependencies.{section}", name, version))

    groups = data.get("dependency-groups", {})
    for section, raw_dependencies in groups.items():
        for raw in dependency_strings(raw_dependencies):
            name, version = extract_name_and_version(raw)
            dependencies.append((f"dependency-groups.{section}", name, version))

    for raw in dependency_strings(data.get("build-system", {}).get("requires", [])):
        name, version = extract_name_and_version(raw)
        dependencies.append(("build-system.requires", name, version))

    errors: list[str] = []
    for section, name, version in dependencies:
        if normalize(name) in source_pinned:
            continue
        if version is None:
            errors.append(f"  [{section}] {name}: no version specified")
        elif not EXACT_VERSION_RE.match(version):
            errors.append(f'  [{section}] {name}: "{version}" is not an exact pin')

    if errors:
        print("Dependency version pinning validation failed.\n")
        print("The following dependencies do not use exact version pins:\n")
        for error in errors:
            print(error)
        print(f"\nTotal: {len(errors)} unpinned dependencies")
        print("All dependencies must use ==X.Y.Z exact versions.")
        print("Packages with git/path/url sources in [tool.uv.sources] are exempt.")
        return 1

    print(f"All {len(dependencies)} dependencies use exact version pins.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
