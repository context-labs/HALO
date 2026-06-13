from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from engine.code.code_repo import CodeRepo


def _build_repo(tmp_path: Path) -> Path:
    """Build a small repo fixture: nested dirs, excluded dirs, binary, long line, escaping symlink."""
    root = tmp_path / "repo"
    (root / "engine" / "tools").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "__pycache__").mkdir()
    (root / "sub").mkdir()

    (root / "engine" / "config.py").write_text('CONFIG = {"max_retries": 3}\n')
    (root / "engine" / "tools" / "runner.py").write_text(
        "import os\n\n\ndef launch():\n    return retries\n"
    )
    (root / "engine" / "main.py").write_text("def main():\n    return 0\n")
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / "__pycache__" / "x.pyc").write_text("cached\n")
    (root / "engine" / "blob.bin").write_bytes(b"\x00\x01binary\x00data\n")
    (root / "engine" / "long.py").write_text('x = "' + "A" * 3000 + '"\n')

    outside = tmp_path / "outside_secret.txt"
    outside.write_text("SECRET\n")
    (root / "sub" / "escape_link").symlink_to(outside)
    return root


# --- open / validation -------------------------------------------------------


def test_open_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CodeRepo.open(tmp_path / "does-not-exist")


def test_open_file_not_directory_raises(tmp_path: Path) -> None:
    f = tmp_path / "afile.txt"
    f.write_text("hi\n")
    with pytest.raises(NotADirectoryError):
        CodeRepo.open(f)


def test_open_resolves_root(tmp_path: Path) -> None:
    root = _build_repo(tmp_path)
    repo = CodeRepo.open(root)
    assert repo.root == root.resolve()


# --- path confinement --------------------------------------------------------


def test_read_parent_traversal_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    with pytest.raises(ValueError, match="outside the repo root"):
        repo.read("../outside_secret.txt", 1, 10)


def test_read_absolute_outside_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    with pytest.raises(ValueError, match="outside the repo root"):
        repo.read("/etc/hosts", 1, 10)


def test_read_symlink_escape_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    with pytest.raises(ValueError, match="outside the repo root"):
        repo.read("sub/escape_link", 1, 10)


def test_read_absolute_inside_root_accepted(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    abs_path = str(repo.root / "engine" / "config.py")
    result = repo.read(abs_path, 1, 10)
    assert result.path == "engine/config.py"


# --- glob --------------------------------------------------------------------


def test_glob_matches_nested_excludes_special_dirs(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.glob("**/*.py", 100)
    assert [f.path for f in result.files] == [
        "engine/config.py",
        "engine/long.py",
        "engine/main.py",
        "engine/tools/runner.py",
    ]
    assert result.returned_count == 4
    assert result.has_more is False
    assert all(f.size_bytes > 0 for f in result.files)


def test_glob_caps_with_has_more(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.glob("**/*.py", 2)
    assert [f.path for f in result.files] == ["engine/config.py", "engine/long.py"]
    assert result.returned_count == 2
    assert result.has_more is True


def test_glob_excludes_symlinks(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.glob("**/*", 500)
    assert "sub/escape_link" not in {f.path for f in result.files}


def test_glob_absolute_pattern_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    with pytest.raises(ValueError, match="must be relative"):
        repo.glob("/etc/*.conf", 10)


def test_glob_parent_traversal_pattern_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    with pytest.raises(ValueError, match="must not contain"):
        repo.glob("../*.txt", 10)


# --- grep (pure-Python path) -------------------------------------------------


def _python_repo(tmp_path: Path) -> CodeRepo:
    """A CodeRepo forced onto the pure-Python grep path (no ripgrep)."""
    root = _build_repo(tmp_path).resolve()
    from engine.code.code_repo import _render_tree

    return CodeRepo(root=root, rg_executable=None, tree=_render_tree(root))


def test_grep_python_line_numbers_and_paths(tmp_path: Path) -> None:
    repo = _python_repo(tmp_path)
    result = repo.grep("retries", None, 50)
    assert [(m.path, m.line_number, m.line_text) for m in result.matches] == [
        ("engine/config.py", 1, 'CONFIG = {"max_retries": 3}'),
        ("engine/tools/runner.py", 5, "    return retries"),
    ]
    assert result.returned_match_count == 2
    assert result.has_more is False


def test_grep_python_glob_filter(tmp_path: Path) -> None:
    repo = _python_repo(tmp_path)
    result = repo.grep("max_retries", "engine/*.py", 50)
    assert [(m.path, m.line_number) for m in result.matches] == [("engine/config.py", 1)]


def test_grep_python_invalid_regex_raises(tmp_path: Path) -> None:
    repo = _python_repo(tmp_path)
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        repo.grep("(", None, 50)


def test_grep_python_caps_with_has_more(tmp_path: Path) -> None:
    repo = _python_repo(tmp_path)
    result = repo.grep(".", None, 1)
    assert result.returned_match_count == 1
    assert result.has_more is True


def test_grep_python_skips_binary_and_excluded(tmp_path: Path) -> None:
    repo = _python_repo(tmp_path)
    paths = {m.path for m in repo.grep(".", None, 500).matches}
    assert paths == {
        "engine/config.py",
        "engine/long.py",
        "engine/main.py",
        "engine/tools/runner.py",
    }


def test_grep_python_truncates_long_line(tmp_path: Path) -> None:
    repo = _python_repo(tmp_path)
    result = repo.grep("AAAA", None, 50)
    assert len(result.matches) == 1
    assert "[HALO truncated:" in result.matches[0].line_text


# --- grep (ripgrep path) -----------------------------------------------------


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_grep_ripgrep_parity(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.grep("retries", None, 50)
    assert sorted((m.path, m.line_number) for m in result.matches) == [
        ("engine/config.py", 1),
        ("engine/tools/runner.py", 5),
    ]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_grep_ripgrep_excludes_special_dirs_and_symlinks(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    paths = {m.path for m in repo.grep(".", None, 500).matches}
    assert paths == {
        "engine/config.py",
        "engine/long.py",
        "engine/main.py",
        "engine/tools/runner.py",
    }


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_grep_ripgrep_syntax_error_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    # Rust regex rejects backreferences; rg exits >=2 with a message on stderr.
    with pytest.raises(ValueError, match="grep failed"):
        repo.grep(r"(\w)\1", None, 50)


# --- read --------------------------------------------------------------------


def test_read_numbers_lines(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.read("engine/main.py", 1, 500)
    assert result.content == "     1\tdef main():\n     2\t    return 0"
    assert result.start_line == 1
    assert result.end_line == 2
    assert result.total_line_count == 2
    assert result.truncated is False


def test_read_offset_and_limit_window(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.read("engine/tools/runner.py", 4, 1)
    assert result.content == "     4\tdef launch():"
    assert result.start_line == 4
    assert result.end_line == 4
    assert result.total_line_count == 5
    assert result.truncated is False


def test_read_long_line_truncates(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.read("engine/long.py", 1, 500)
    assert result.truncated is True
    assert "[HALO truncated:" in result.content


def test_read_empty_file(tmp_path: Path) -> None:
    root = _build_repo(tmp_path)
    (root / "empty.txt").write_text("")
    repo = CodeRepo.open(root)
    result = repo.read("empty.txt", 1, 500)
    assert result.content == ""
    assert result.end_line == 0
    assert result.total_line_count == 0
    assert result.truncated is False


def test_read_missing_file_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    with pytest.raises(ValueError, match="not a file"):
        repo.read("engine/nope.py", 1, 10)


def test_read_binary_file_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    with pytest.raises(ValueError, match="binary file"):
        repo.read("engine/blob.bin", 1, 10)


# --- tree --------------------------------------------------------------------


def test_tree_excludes_special_dirs_and_symlinks(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    tree = repo.tree
    assert "engine/" in tree
    assert "config.py" in tree
    assert ".git" not in tree
    assert "__pycache__" not in tree
    assert "escape_link" not in tree


def test_tree_depth_cap_marker(tmp_path: Path) -> None:
    root = tmp_path / "deep"
    deep = root / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "buried.py").write_text("x = 1\n")
    repo = CodeRepo.open(root)
    assert "depth cap reached" in repo.tree


def test_tree_entry_cap_marker(tmp_path: Path) -> None:
    root = tmp_path / "wide"
    root.mkdir()
    for i in range(600):
        (root / f"f{i:04d}.txt").write_text("x\n")
    repo = CodeRepo.open(root)
    assert "entry cap of 500 reached" in repo.tree
