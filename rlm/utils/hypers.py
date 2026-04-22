"""Dataclass-based configuration with automatic CLI generation.

Define a config as a dataclass inheriting from Hypers. Every field becomes a
CLI argument automatically. Values are resolved as: defaults → CLI args.

Usage::

    @dataclass
    class MyConfig(Hypers):
        model: str = "gpt-4o"
        temperature: float = 0.0
        workers: int = 8
        max_rps: float = 10.0

    # Just defaults:
    #   uv run my-project generate

    # With a CLI override:
    #   uv run my-project generate --temperature 0.5

Vendored from https://github.com/vmasrani/machine_learning_helpers; the
config-file loading path was removed because nobody used it and scanning
argv for ``*.py`` tokens led to arbitrary code execution when other CLIs
(pytest, modal) happened to pass file paths through.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import Field, dataclass, field, fields
from typing import Any, get_args, get_origin, get_type_hints

from rich.console import Console
from rich.table import Table

_console = Console()


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
    """Base class for configuration dataclasses with automatic CLI generation.

    Subclass this with your config fields. On instantiation, fields are set
    in order: defaults → CLI args. CLI overrides everything.

    The ``__str__`` method displays a color-coded table showing where each
    value came from (default vs. CLI).

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
        """Apply the layered override: defaults → CLI args."""
        self._raise_untyped()

        changed_args = self._init_argparse()

        # Track where each value came from for display.
        cli_names = {k for k, _ in changed_args}
        self._source: dict[str, str] = {}
        for f in self._all_fields():
            self._source[f.name] = "cli" if f.name in cli_names else "default"

        # Apply CLI values (highest priority).
        for k, v in changed_args:
            self.set(k, v)

    def _init_argparse(self) -> list[tuple[str, Any]]:
        """Build an argparse parser from dataclass fields and parse sys.argv.

        Only dataclass fields become CLI args; anything else in argv (e.g.
        ``pytest`` test-file paths, ``modal run`` entrypoint paths) is
        ignored via ``parse_known_args``.

        Returns:
            List of ``(name, value)`` pairs for fields the caller
            explicitly passed on the command line. Fields left at their
            dataclass defaults don't appear here.
        """
        parser = argparse.ArgumentParser(allow_abbrev=False)
        hints = get_type_hints(type(self))
        for f in self._all_fields():
            _add_to_parser(parser, f.name, self.get(f.name), hints.get(f.name))

        args, _ = parser.parse_known_args()
        # "Explicit" means the flag was literally present in sys.argv.
        # argparse alone can't tell — it fills in the default either way.
        changed_args: list[tuple[str, Any]] = []
        field_names = {f.name for f in self._all_fields()}
        for raw in sys.argv[1:]:
            if not raw.startswith("--"):
                continue
            key = raw.removeprefix("--").split("=", 1)[0]
            if key in field_names and hasattr(args, key):
                changed_args.append((key, getattr(args, key)))
        return changed_args

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

        Color coding: [blue]default[/], [yellow]CLI override[/].
        """
        table = Table(title="Config", show_header=True, header_style="bold")
        table.add_column("Parameter", style="dim")
        table.add_column("Value")
        table.add_column("Source")

        source = getattr(self, "_source", {})
        style_map = {"default": "blue", "cli": "yellow"}

        for f in self._all_fields():
            name = f.name
            val = str(self.get(name))
            src = source.get(name, "default")
            style = style_map.get(src, "blue")
            table.add_row(name, f"[{style}]{val}[/]", f"[{style}]{src}[/]")

        with _console.capture() as capture:
            _console.print(table)
        return capture.get()
