"""Dataclass-based configuration with automatic CLI generation and config file support.

Define a config as a dataclass inheriting from Hypers. Every field becomes a CLI
argument automatically. Config files (plain Python) can override defaults, and
CLI arguments override everything.

Layer order: defaults → config file → CLI args

Usage::

    @dataclass
    class MyConfig(Hypers):
        model: str = "gpt-4o"
        temperature: float = 0.0
        workers: int = 8
        max_rps: float = 10.0

    # Just defaults:
    #   uv run my-project generate

    # With config file override:
    #   uv run my-project generate configs/fast.py

    # With config file + CLI override:
    #   uv run my-project generate configs/fast.py --temperature 0.5

Config files are plain Python with variable assignments::

    # configs/fast.py
    model = "gpt-4o-mini"
    temperature = 0.2
    workers = 16

Vendored from https://github.com/vmasrani/machine_learning_helpers
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import Field, dataclass, field, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from rich.console import Console
from rich.table import Table

_console = Console()


def _is_notebook() -> bool:
    """Check if code is running inside a Jupyter notebook."""
    try:
        from IPython.core.getipython import get_ipython  # type: ignore[import-not-found]

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _resolve_type(type_hint: Any) -> type | None:
    """Resolve the concrete type from a type hint, handling Optional/Union with None.

    Handles both ``Optional[X]`` and ``X | None`` syntax by extracting the
    non-None type argument.

    Args:
        type_hint: The resolved type hint (not a string — use typing.get_type_hints first).

    Returns:
        The resolved type, or None if it cannot be determined.
    """
    origin = get_origin(type_hint)
    if origin is not None:
        args = [a for a in get_args(type_hint) if a is not type(None)]
        return args[0] if args else None
    if isinstance(type_hint, type):
        return type_hint
    return None


def _parse_bool(value: str) -> bool:
    """Parse a string into a boolean value.

    Args:
        value: String to parse (accepts yes/true/t/y/1/no/false/f/n/0).

    Returns:
        The parsed boolean.

    Raises:
        ValueError: If the string is not a recognized boolean value.
    """
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Cannot parse {value!r} as boolean")


def _add_to_parser(
    parser: argparse.ArgumentParser, name: str, val: object, resolved_hint: type | None
) -> None:
    """Add a dataclass field as an argparse argument.

    Args:
        parser: The argument parser to add to.
        name: The field name (becomes --name).
        val: The current default value.
        resolved_hint: The resolved type hint (from get_type_hints), used when val is None.
    """
    if isinstance(val, bool):
        parser.add_argument(f"--{name}", type=_parse_bool, default=val)
    elif val is None:
        concrete = _resolve_type(resolved_hint) if resolved_hint is not None else None
        if concrete is not None:
            parser.add_argument(f"--{name}", type=concrete, default=None)
        else:
            parser.add_argument(f"--{name}", default=None)
    else:
        parser.add_argument(f"--{name}", type=type(val), default=val)


def read_config(file: str | Path) -> dict[str, Any]:
    """Read a Python config file and return its variables as a dict.

    Config files are plain Python files with variable assignments.
    Variables starting with '_' are ignored.

    Args:
        file: Path to the Python config file.

    Returns:
        Dict of variable names to values.
    """
    variables: dict[str, Any] = {}
    with open(file, encoding="utf-8") as f:
        exec(f.read(), variables)  # noqa: S102
    return {k: v for k, v in variables.items() if not k.startswith("_")}


def TBD(default: Any = None) -> Any:  # noqa: N802
    """Mark a field as "to be determined" — excluded from CLI args and __init__.

    Use this for fields that are computed or set programmatically after init.

    Args:
        default: Optional default value. Can be None, a scalar, or a list.

    Returns:
        A dataclass field configured with init=False, repr=False.
    """
    if default is None:
        return field(init=False, repr=False)
    if isinstance(default, list):
        default_copy: Any = default
        return field(default_factory=lambda: list(default_copy), init=False, repr=False)
    return field(default=default, init=False, repr=False)


@dataclass(repr=False)
class Hypers:
    """Base class for configuration dataclasses with CLI and config file support.

    Subclass this with your config fields. On instantiation, fields are set
    in order: defaults → config file → CLI args. Each layer overrides the previous.

    The ``__str__`` method displays a color-coded table showing where each value
    came from (default, config file, or CLI).

    Example::

        @dataclass
        class TrainingConfig(Hypers):
            learning_rate: float = 1e-4
            batch_size: int = 32
            epochs: int = 10
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Remove dataclass-generated __repr__ so our Rich table is used."""
        super().__init_subclass__(**kwargs)
        if "__repr__" in cls.__dict__:
            delattr(cls, "__repr__")

    def __post_init__(self) -> None:
        """Apply the layered override: defaults → config file → CLI args."""
        self._raise_untyped()

        argv, changed_args = self._init_argparse()
        file_vars = self._parse_config_files(argv)

        # Track where each value came from for display
        cli_names = {k for k, _ in changed_args}
        config_names = {arg for var in file_vars.values() for arg in var}
        self._source: dict[str, str] = {}
        for f in self._all_fields():
            if f.name in cli_names:
                self._source[f.name] = "cli"
            elif f.name in config_names:
                self._source[f.name] = "config"
            else:
                self._source[f.name] = "default"

        # Apply config file values
        for variables in file_vars.values():
            for name, value in variables.items():
                self.set(name, value)

        # Apply CLI values (highest priority)
        for k, v in changed_args:
            self.set(k, v)

    def _init_argparse(self) -> tuple[list[str], list[tuple[str, Any]]]:
        """Build an argparse parser from dataclass fields and parse sys.argv.

        Returns:
            Tuple of (remaining argv not consumed by argparse, list of (name, value)
            pairs for args explicitly passed on the command line).
        """
        parser = argparse.ArgumentParser(allow_abbrev=False)
        hints = get_type_hints(type(self))
        for f in self._all_fields():
            _add_to_parser(parser, f.name, self.get(f.name), hints.get(f.name))

        args, argv = parser.parse_known_args()
        keys = [arg.replace("--", "").split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")]
        changed_args = [(k, getattr(args, k)) for k in keys if hasattr(args, k)]
        return argv, changed_args

    def _parse_config_files(self, argv: list[str]) -> dict[str, dict[str, Any]]:
        """Extract and parse .py config files from the remaining argv.

        Args:
            argv: Remaining command line arguments after argparse consumed known args.

        Returns:
            Dict mapping config file paths to their parsed variables.
        """
        if _is_notebook():
            return {}
        configs = [f for f in argv if f.endswith(".py")]
        return {cfg: read_config(cfg) for cfg in configs}

    def _all_fields(self) -> list[Field[Any]]:
        """Return all dataclass fields that participate in __init__."""
        return [f for f in fields(self) if f.init]

    def _all_variables(self) -> list[str]:
        """Return all class-level variable names (excludes private/dunder, methods, descriptors)."""
        return [
            n
            for n, v in self.__class__.__dict__.items()
            if not n.startswith("_") and not callable(v) and not isinstance(v, property)
        ]

    def _raise_untyped(self) -> None:
        """Raise ValueError if any class variables lack type annotations."""
        all_vars = set(self._all_variables())
        all_field_names = {f.name for f in fields(self)}  # ALL fields, including init=False (TBD)
        untyped = all_vars - all_field_names
        if untyped:
            raise ValueError(f"Variables missing type annotations: {', '.join(untyped)}")

    def get(self, name: str) -> Any:
        """Get a config value by name.

        Args:
            name: The field name.

        Returns:
            The current value.
        """
        return getattr(self, name)

    def set(self, name: str, val: Any) -> None:
        """Set a config value by name.

        Args:
            name: The field name.
            val: The new value.
        """
        setattr(self, name, val)

    def to_dict(self) -> dict[str, Any]:
        """Return all config values as a dict.

        Returns:
            Dict of field names to values (includes TBD/computed fields).
        """
        return {f.name: self.get(f.name) for f in fields(self)}

    def update(self, new_dict: dict[str, Any]) -> None:
        """Update multiple config values from a dict.

        Args:
            new_dict: Dict of field names to new values.
        """
        for k, v in new_dict.items():
            self.set(k, v)

    def __repr__(self) -> str:
        """Render a Rich table showing config values and their sources."""
        return self.__str__()

    def __str__(self) -> str:
        """Render a Rich table showing config values and their sources.

        Color coding: [blue]default[/], [magenta]config file[/], [yellow]CLI override[/].
        """
        table = Table(title="Config", show_header=True, header_style="bold")
        table.add_column("Parameter", style="dim")
        table.add_column("Value")
        table.add_column("Source")

        source = getattr(self, "_source", {})
        style_map = {"default": "blue", "config": "magenta", "cli": "yellow"}

        for f in self._all_fields():
            name = f.name
            val = str(self.get(name))
            src = source.get(name, "default")
            style = style_map.get(src, "blue")
            table.add_row(name, f"[{style}]{val}[/]", f"[{style}]{src}[/]")

        with _console.capture() as capture:
            _console.print(table)
        return capture.get()
