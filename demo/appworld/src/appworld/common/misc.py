import difflib
import json
import os
import re
import subprocess
from subprocess import CalledProcessError
from typing import Any, Literal, cast

from rich.console import Console

from appworld.common.collections import getter_plus
from appworld.common.constants import DIFFICULTY_NUM_CALLS
from appworld.common.imports import import_apis_module
from appworld.common.io import read_file
from appworld.common.types import TItem


def get_difficulty_level(num_requests: int) -> str:
    for difficulty, (lower_bound, upper_bound) in DIFFICULTY_NUM_CALLS.items():
        if lower_bound <= num_requests <= upper_bound:
            print(f"****************************** {num_requests} >> {difficulty}")
            return difficulty
    return "unknown"


def get_login_by(app_name: str) -> str | None:
    return cast(str | None, import_apis_module(app_name).login_by)


def show_line_diff(text_1: str, text_2: str, show_unchanged: bool = False) -> bool:
    """Shows git-styled line diff between two strings."""

    lines1 = text_1.splitlines(keepends=True)
    lines2 = text_2.splitlines(keepends=True)
    differ = difflib.Differ()
    diff = list(differ.compare(lines1, lines2))
    console = Console()
    changed = False
    for line in diff:
        if show_unchanged and line.startswith("  "):
            console.print(line.rstrip(), style="white")
        if line.startswith("- "):
            console.print(line.rstrip(), style="red")
            changed = True
        if line.startswith("+ "):
            console.print(line.rstrip(), style="green")
            changed = True
    return changed


def rerank_by_similarity(
    items: list[TItem], query: str, field: str, similarity_method: Literal["semantic", "string"]
) -> list[TItem]:
    query = " ".join(query.lower().split())

    def similarity_score(item: TItem) -> float:
        if similarity_method == "string":
            # TODO: Use rapidfuzz if this function is heavily used.
            from difflib import SequenceMatcher

            value = getter_plus(item, field)
            return SequenceMatcher(None, query, value).ratio()
        if similarity_method == "semantic":
            raise NotImplementedError("Semantic similarity is not implemented yet.")

    return sorted(items, key=similarity_score, reverse=True)


def get_version(with_git_hash: bool = False) -> str:
    import appworld

    version = appworld.__version__
    if with_git_hash:
        project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            output = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=project_directory,
                stderr=subprocess.STDOUT,
            )
            if not output:
                return version
            git_hash = output.decode("utf-8").strip()
            version += f":{git_hash}"
        except (CalledProcessError, FileNotFoundError, NotADirectoryError, PermissionError):
            pass
    return version


def parse_project_version(pyproject_path: str) -> str | None:
    if not os.path.exists(pyproject_path):
        raise FileNotFoundError(f"{pyproject_path} does not exist.")
    text = cast(str, read_file(pyproject_path))
    in_project = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            in_project = section == "project"
            continue
        if in_project and line.startswith("version"):
            match = re.match(r"version\s*=\s*(['\"])(.*?)\1", line)
            if match:
                return match.group(2).strip()
            if "=" in line:
                value = line.split("=", 1)[1].split("#", 1)[0].strip()
                if value and (value[0] == value[-1] and value[0] in ('"', "'")):
                    return value[1:-1].strip()
                if value:
                    return value
    return None


def build_cli_command(
    command_name: str,
    arguments: dict[str, Any],
    use_hyphens: bool = True,
) -> str:
    command = f"{command_name} "
    for key, value in arguments.items():
        if use_hyphens:
            key = key.replace("_", "-")
        if value is True:
            command += f"--{key} "
        elif value is False:
            continue
        elif isinstance(value, list):
            command += f"--{key} " + " ".join(json.dumps(v) for v in value) + " "
        else:
            command += f"--{key} {json.dumps(value)} "
    return command.strip()
