from __future__ import annotations

from pathlib import Path

import pytest

from engine.code.code_repo import CodeRepo
from engine.errors import EngineDependencyError


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


def _build_gitignore_repo(tmp_path: Path) -> Path:
    """A repo whose .gitignore excludes a dir and a glob NOT in the fixed excluded-dirs set."""
    root = tmp_path / "gi"
    (root / "src").mkdir(parents=True)
    (root / "ignored_dir").mkdir()
    (root / "src" / "app.py").write_text("MARK = 1\n")
    (root / "keep.py").write_text("MARK = 2\n")
    (root / "debug.log").write_text("MARK\n")
    (root / "ignored_dir" / "x.py").write_text("MARK = 3\n")
    (root / ".gitignore").write_text("*.log\nignored_dir/\n")
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


def test_open_raises_without_ripgrep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.code.code_repo as code_repo_module

    monkeypatch.setattr(code_repo_module, "find_ripgrep", lambda: None)
    with pytest.raises(EngineDependencyError, match="ripgrep"):
        CodeRepo.open(_build_repo(tmp_path))


# --- path confinement (read) -------------------------------------------------


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


def test_glob_anchored_pattern(tmp_path: Path) -> None:
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


def test_glob_star_matches_any_depth(tmp_path: Path) -> None:
    """gitignore-style: a pattern without `/` matches at any depth."""
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.glob("*.py", 100)
    assert [f.path for f in result.files] == [
        "engine/config.py",
        "engine/long.py",
        "engine/main.py",
        "engine/tools/runner.py",
    ]


def test_glob_caps_with_has_more(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.glob("**/*.py", 2)
    assert [f.path for f in result.files] == ["engine/config.py", "engine/long.py"]
    assert result.returned_count == 2
    assert result.has_more is True


def test_glob_excludes_special_dirs_and_symlinks(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    paths = {f.path for f in repo.glob("**/*", 500).files}
    assert "sub/escape_link" not in paths
    assert not any(p.startswith(".git/") or "__pycache__" in p for p in paths)


def test_glob_honors_gitignore(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_gitignore_repo(tmp_path))
    paths = {f.path for f in repo.glob("*.py", 100).files}
    assert paths == {"keep.py", "src/app.py"}  # ignored_dir/x.py excluded by .gitignore


def test_glob_skips_unstattable_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A path rg listed but that vanished before stat (a race) is skipped, not fatal."""
    repo = CodeRepo.open(_build_repo(tmp_path))

    def _fake_stream(_args: list[str]):
        yield "engine/config.py\n"
        yield "engine/ghost.py\n"  # listed by rg, but not on disk at stat time

    monkeypatch.setattr(repo, "_rg_line_stream", _fake_stream)
    result = repo.glob("**/*.py", 100)
    assert [f.path for f in result.files] == ["engine/config.py"]


# --- grep --------------------------------------------------------------------


def test_grep_line_numbers_and_paths(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.grep("retries", None, 50)
    assert sorted((m.path, m.line_number) for m in result.matches) == [
        ("engine/config.py", 1),
        ("engine/tools/runner.py", 5),
    ]
    assert result.has_more is False


def test_grep_glob_filter(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.grep("max_retries", "engine/*.py", 50)
    assert [(m.path, m.line_number) for m in result.matches] == [("engine/config.py", 1)]


def test_grep_caps_with_has_more(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.grep(".", None, 1)
    assert result.returned_match_count == 1
    assert result.has_more is True


def test_grep_excludes_special_dirs_and_binary(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    paths = {m.path for m in repo.grep(".", None, 500).matches}
    assert paths == {
        "engine/config.py",
        "engine/long.py",
        "engine/main.py",
        "engine/tools/runner.py",
    }


def test_grep_honors_gitignore(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_gitignore_repo(tmp_path))
    paths = {m.path for m in repo.grep("MARK", None, 500).matches}
    assert paths == {"keep.py", "src/app.py"}  # debug.log + ignored_dir/ excluded


def test_grep_invalid_regex_raises(tmp_path: Path) -> None:
    repo = CodeRepo.open(_build_repo(tmp_path))
    # Rust regex rejects backreferences; rg exits >=2 with a message on stderr.
    with pytest.raises(ValueError, match="ripgrep failed"):
        repo.grep(r"(\w)\1", None, 50)


def _fake_rg(tmp_path: Path, script: str) -> CodeRepo:
    """A CodeRepo whose 'ripgrep' is a stub shell script, for exit-code edge cases."""
    fake = tmp_path / "fake-rg"
    fake.write_text(f"#!/bin/sh\n{script}\n")
    fake.chmod(0o755)
    return CodeRepo(root=tmp_path.resolve(), rg_executable=str(fake))


def test_rg_stream_keeps_partial_output_on_error(tmp_path: Path) -> None:
    """rg exiting >=2 *after* emitting output keeps the partial output, not raises."""
    repo = _fake_rg(tmp_path, "printf 'a/b.py\\n'\nexit 2")
    assert list(repo._rg_line_stream(repo._rg_files_args(glob_pattern=None, sort=False))) == [
        "a/b.py\n"
    ]


def test_rg_stream_raises_on_error_without_output(tmp_path: Path) -> None:
    """rg exiting >=2 with no output (e.g. a bad pattern) raises with its stderr."""
    repo = _fake_rg(tmp_path, "echo boom 1>&2\nexit 2")
    with pytest.raises(ValueError, match="ripgrep failed"):
        list(repo._rg_line_stream(repo._rg_files_args(glob_pattern=None, sort=False)))


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


def test_read_long_line_does_not_drop_later_lines(tmp_path: Path) -> None:
    """A capped long line must not suppress the shorter lines after it (budget remains)."""
    root = _build_repo(tmp_path)
    (root / "mixed.py").write_text('x = "' + "A" * 3000 + '"\nshort = 1\nalso = 2\n')
    repo = CodeRepo.open(root)
    result = repo.read("mixed.py", 1, 500)
    lines = result.content.split("\n")
    assert lines[0].startswith("     1\t") and "[HALO truncated:" in lines[0]
    assert lines[1] == "     2\tshort = 1"
    assert lines[2] == "     3\talso = 2"
    assert result.start_line == 1
    assert result.end_line == 3
    assert result.total_line_count == 3
    assert result.truncated is True


def test_read_empty_file(tmp_path: Path) -> None:
    root = _build_repo(tmp_path)
    (root / "empty.txt").write_text("")
    repo = CodeRepo.open(root)
    result = repo.read("empty.txt", 1, 500)
    assert result.content == ""
    assert result.start_line == 0
    assert result.end_line == 0
    assert result.total_line_count == 0
    assert result.truncated is False


def test_read_offset_past_eof(tmp_path: Path) -> None:
    """An offset beyond the last line yields an empty, non-contradictory window."""
    repo = CodeRepo.open(_build_repo(tmp_path))
    result = repo.read("engine/main.py", 99, 10)  # main.py has 2 lines
    assert result.content == ""
    assert result.start_line == 0
    assert result.end_line == 0
    assert result.total_line_count == 2
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
    # The whole tree: dirs first then files (alphabetical), binaries listed, but
    # no .git/__pycache__ (excluded) and no sub/escape_link (symlink, not followed).
    repo = CodeRepo.open(_build_repo(tmp_path))
    assert repo.tree == (
        "repo/\n"
        "  engine/\n"
        "    tools/\n"
        "      runner.py\n"
        "    blob.bin\n"
        "    config.py\n"
        "    long.py\n"
        "    main.py"
    )


def test_tree_honors_gitignore(tmp_path: Path) -> None:
    # debug.log and ignored_dir/ are gitignored and absent; .gitignore itself is a
    # real (hidden) file and is listed.
    repo = CodeRepo.open(_build_gitignore_repo(tmp_path))
    assert repo.tree == ("gi/\n  src/\n    app.py\n  .gitignore\n  keep.py")


def test_tree_depth_cap_marker(tmp_path: Path) -> None:
    root = tmp_path / "deep"
    deep = root / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "buried.py").write_text("x = 1\n")
    repo = CodeRepo.open(root)
    assert repo.tree == ("deep/\n  a/\n    b/\n      c/\n        ... (depth cap reached)")


def test_tree_entry_cap_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.code.code_repo as code_repo_module

    monkeypatch.setattr(code_repo_module, "_TREE_MAX_ENTRIES", 2)
    root = tmp_path / "ec"
    root.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (root / name).write_text("x\n")
    repo = CodeRepo.open(root)
    assert repo.tree == ("ec/\n  a.py\n  b.py\n  ... (entry cap of 2 reached)")


def test_tree_caps_input_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tree reads at most _TREE_MAX_PATHS (sorted) paths, not the whole repo."""
    import engine.code.code_repo as code_repo_module

    monkeypatch.setattr(code_repo_module, "_TREE_MAX_PATHS", 2)
    root = tmp_path / "many"
    root.mkdir()
    for name in ("a.py", "b.py", "c.py", "d.py"):
        (root / name).write_text("x\n")
    repo = CodeRepo.open(root)
    # Only the two lexicographically-first paths are read, so c.py/d.py never appear.
    assert repo.tree == ("many/\n  a.py\n  b.py")


def test_tree_is_cached(tmp_path: Path) -> None:
    """The tree is rendered lazily on first access and memoized for the run."""
    repo = CodeRepo.open(_build_repo(tmp_path))
    first = repo.tree
    # Same object returned on subsequent access — rendered once, then cached.
    assert repo.tree is first


def test_tree_not_rendered_at_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``open`` must not render the tree — it is deferred to first ``view_repo_tree``."""
    import engine.code.code_repo as code_repo_module

    calls = {"n": 0}
    real_build = code_repo_module._build_tree

    def _counting_build(root_name: str, paths: list[str]) -> str:
        calls["n"] += 1
        return real_build(root_name, paths)

    monkeypatch.setattr(code_repo_module, "_build_tree", _counting_build)
    repo = CodeRepo.open(_build_repo(tmp_path))
    assert calls["n"] == 0
    _ = repo.tree
    assert calls["n"] == 1
    _ = repo.tree
    assert calls["n"] == 1
