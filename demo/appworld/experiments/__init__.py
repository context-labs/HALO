import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_distribution_version

from appworld.common.misc import parse_project_version


_version: str | None
try:
    _version = get_distribution_version("appworld-agents")
except PackageNotFoundError:
    file_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    _version = parse_project_version(file_path)

if _version is None:
    raise RuntimeError("Could not determine the version of the appworld-agents package.")
assert _version is not None
__version__ = _version

__all__ = ["__version__"]
