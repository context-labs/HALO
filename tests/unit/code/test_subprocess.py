from __future__ import annotations

import sys
from pathlib import Path

import pytest

from engine.code._subprocess import stream_subprocess_lines


def _py(code: str) -> list[str]:
    """An argv that runs ``code`` under this interpreter — deterministic flush control."""
    return [sys.executable, "-c", code]


def test_yields_stdout_lines(tmp_path: Path) -> None:
    lines = list(
        stream_subprocess_lines(
            _py("print('a')\nprint('b')\nprint('c')"),
            cwd=tmp_path,
            error_label="prog",
        )
    )
    assert lines == ["a\n", "b\n", "c\n"]


def test_early_stop_terminates_without_error(tmp_path: Path) -> None:
    """Closing the generator after a partial read terminates a still-running process, no error."""
    gen = stream_subprocess_lines(
        _py("import time\nprint('a', flush=True)\ntime.sleep(30)"),
        cwd=tmp_path,
        error_label="prog",
    )
    assert next(gen) == "a\n"
    # If terminate-on-GeneratorExit were broken, this would hang on the sleep.
    gen.close()


def test_large_stderr_does_not_deadlock(tmp_path: Path) -> None:
    """A process that writes >64KB to stderr before stdout must not deadlock (concurrent drain)."""
    code = (
        "import sys\n"
        "sys.stderr.write('E' * (1024 * 1024))\n"
        "sys.stderr.flush()\n"
        "for i in range(100):\n"
        "    print('out', i)\n"
    )
    lines = list(stream_subprocess_lines(_py(code), cwd=tmp_path, error_label="prog"))
    assert len(lines) == 100
    assert lines[0] == "out 0\n"
    assert lines[-1] == "out 99\n"


def test_raises_on_error_without_output(tmp_path: Path) -> None:
    code = "import sys\nsys.stderr.write('boom')\nsys.exit(2)"
    with pytest.raises(ValueError, match="prog failed: boom"):
        list(stream_subprocess_lines(_py(code), cwd=tmp_path, error_label="prog"))


def test_keeps_partial_output_on_error(tmp_path: Path) -> None:
    """Exiting non-zero AFTER emitting output keeps the partial output rather than raising."""
    code = "import sys\nprint('kept', flush=True)\nsys.exit(2)"
    lines = list(stream_subprocess_lines(_py(code), cwd=tmp_path, error_label="prog"))
    assert lines == ["kept\n"]


def test_returncode_floor_below_floor_is_not_error(tmp_path: Path) -> None:
    """Exit code below the floor (e.g. ripgrep's 1 = 'no matches') is not an error."""
    code = "import sys\nsys.stderr.write('no matches')\nsys.exit(1)"
    lines = list(
        stream_subprocess_lines(
            _py(code), cwd=tmp_path, error_label="prog", error_returncode_floor=2
        )
    )
    assert lines == []


def test_returncode_floor_at_floor_is_error(tmp_path: Path) -> None:
    """The same exit 1 IS an error under floor=1 (git's bad-ref convention)."""
    code = "import sys\nsys.stderr.write('bad ref')\nsys.exit(1)"
    with pytest.raises(ValueError, match="prog failed: bad ref"):
        list(
            stream_subprocess_lines(
                _py(code), cwd=tmp_path, error_label="prog", error_returncode_floor=1
            )
        )
