from __future__ import annotations

import importlib
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from enum import Enum
from io import StringIO
from multiprocessing import Process, Queue
from textwrap import dedent, indent
from typing import Any, Literal, cast

import typer
from click import Context
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from typer.core import TyperGroup

import appworld
from appworld import update_root
from appworld.common.background_server import BackgroundServer
from appworld.common.collections import non_none, override_dict
from appworld.common.constants import (
    DEFAULT_REMOTE_APIS_PORT,
    DEFAULT_REMOTE_APIS_URL,
    DEFAULT_REMOTE_ENVIRONMENT_PORT,
    DEFAULT_REMOTE_MCP_PORT,
    DEFAULT_REMOTE_UIS_PORT,
)
from appworld.common.imports import ensure_package_installed
from appworld.common.io import TERMINAL_LOGGER_WIDTH, write_json
from appworld.common.misc import build_cli_command
from appworld.common.path_store import path_store
from appworld.common.printer import print_rule, rprint
from appworld.common.system import get_available_port


class OrderCommands(TyperGroup):
    # https://github.com/tiangolo/typer/issues/428
    def list_commands(self, ctx: Context) -> list[str]:
        """Return list of commands in the order appear."""
        ordered_command_keys = [
            "install",
            "download",
            "verify",
            "explore",
            "play",
            "run",
            "evaluate",
            "serve",
            "pack",
            "unpack",
            "make",
        ]
        if sorted(ordered_command_keys) != sorted(self.commands.keys()):
            raise Exception(
                "The commands in the appworld.cli.py are not matching the "
                "expected ones in the 'ordered_command_keys' list in the OrderCommands class. "
                "Please add missing and remove extra commands to/from the list."
            )
        return ordered_command_keys


class ServeOrderCommands(TyperGroup):
    def list_commands(self, ctx: Context) -> list[str]:
        ordered = [
            "environment",
            "apis",
            "mcp",
            # "uis",  # TODO: Uncomment with the next release
            "multiple",
        ]
        if sorted(ordered) != sorted(self.commands.keys()):
            raise Exception(
                "The sub-commands in the serve group do not match the expected set. "
                f"Expected: {ordered}. Found: {sorted(self.commands.keys())}. "
                "Please add missing and remove extra commands to/from the list."
            )
        return ordered


class DownloadOrderCommands(TyperGroup):
    def list_commands(self, ctx: Context) -> list[str]:
        ordered = ["data", "experiment-outputs"]
        if sorted(ordered) != sorted(self.commands.keys()):
            raise Exception(
                "The sub-commands in the serve group do not match the expected set. "
                f"Expected: {ordered}. Found: {sorted(self.commands.keys())}. "
                "Please add missing and remove extra commands to/from the list."
            )
        return ordered


class VerifyOrderCommands(TyperGroup):
    def list_commands(self, ctx: Context) -> list[str]:
        ordered = ["tests", "tasks"]
        if sorted(ordered) != sorted(self.commands.keys()):
            raise Exception(
                "The sub-commands in the serve group do not match the expected set. "
                f"Expected: {ordered}. Found: {sorted(self.commands.keys())}. "
                "Please add missing and remove extra commands to/from the list."
            )
        return ordered


app = typer.Typer(
    help="AppWorld Command Line Interface 🚀",
    add_completion=False,
    cls=OrderCommands,
)


class DownloadEntity(str, Enum):
    data = "data"
    experiment_outputs = "experiment-outputs"


class DatasetName(str, Enum):
    test_normal = "test_normal"
    test_challenge = "test_challenge"


class VerifyEntity(str, Enum):
    tests = "tests"
    tasks = "tasks"


class LeaderboardCommand(str, Enum):
    prepare = "prepare"
    submit = "submit"


class ServerType(str, Enum):
    multiple = "multiple"
    environment = "environment"
    apis = "apis"
    mcp = "mcp"
    uis = "uis"


class Transport(str, Enum):
    stdio = "stdio"
    http = "http"


class ExtendedDatasetName(str, Enum):
    train = "train"
    dev = "dev"
    test_normal = "test_normal"
    test_challenge = "test_challenge"


class Difficulty(str, Enum):
    one = 1
    two = 2
    three = 3


class NumTasksPerScenario(str, Enum):
    one = 1
    two = 2
    three = 3


class MCPOutputType(str, Enum):
    content_only = "content_only"
    structured_data_only = "structured_data_only"
    both_but_empty_text = "both_but_empty_text"
    both = "both"
    call_tool_result = "call_tool_result"


class DataDownloadMode(str, Enum):
    minimal = "minimal"
    partial = "partial"
    full = "full"


def appworld_installation_type() -> Literal["repo", "package"]:
    return "package" if "site-packages" in appworld.__path__[0] else "repo"


def appworld_installation_command() -> str:
    command = "appworld install"
    if appworld_installation_type() == "repo":
        command += " --repo"
    return command


def verify_data_exists(with_setup: bool = False) -> None:
    if os.path.exists(path_store.data):
        return
    if with_setup:
        from appworld.download import download_data

        print(
            "Did not find any ./data in the AppWorld's root directory "
            f"which is set to '{os.path.abspath(path_store.root)}'. "
            "Running 'appworld download data'..."
        )
        download_data()
        return
    raise Exception(
        "Did not find any ./data in the AppWorld's root directory "
        f"which is set to '{os.path.abspath(path_store.root)}'. Either download it "
        "manually (e.g., with 'appworld download data') or pass the "
        "--with-setup flag to download it automatically. Also check that you "
        "have set the AppWorld root directory appropriately. See the README "
        "for more information."
    )


def is_fully_installed() -> bool:
    import appworld

    installed_fully = True
    try:
        _ = appworld.apps.api_docs
    except (AttributeError, ModuleNotFoundError):
        installed_fully = False
    path_store_cache_options = {
        "repo": os.path.dirname(os.path.dirname(appworld.__path__[0])),
        "package": path_store.cache,
    }
    path_store_cache = path_store_cache_options[appworld_installation_type()]
    tests_directory = os.path.join("tests", "package", "apps")
    absolute_tests_directory = os.path.join(path_store_cache, tests_directory)
    if not os.path.exists(absolute_tests_directory):
        installed_fully = False
    return installed_fully


def verify_fully_installed(with_setup: bool = False) -> None:
    if is_fully_installed():
        return
    if with_setup:
        from appworld.install import install_package, install_repo

        command = appworld_installation_command()
        print(f"The appworld package is not fully installed. Running '{command}'...")
        install = {"repo": install_repo, "package": install_package}[appworld_installation_type()]
        install()
        return
    command_suffix = {"repo": " --repo", "package": ""}[appworld_installation_type()]
    raise Exception(
        "The appworld package is not fully installed. Either manually run "
        f"'appworld install{command_suffix}' or 'python -m appworld.cli install{command_suffix}' "
        "to complete the installation, or pass the --with-setup flag to automatically install it."
    )


def load_experiment_config(experiment_name: str) -> dict[str, Any]:
    from appworld.common.io import jsonnet_load

    experiment_file_path = path_store.experiment_config_file_path(experiment_name)
    if not os.path.exists(experiment_file_path):
        raise Exception(f"The experiment config file {experiment_file_path} does not exist.")
    experiment_config = jsonnet_load(
        experiment_file_path,
        APPWORLD_EXPERIMENT_PROMPTS_PATH=path_store.experiment_prompts,
        APPWORLD_EXPERIMENT_CONFIGS_PATH=path_store.experiment_configs,
        APPWORLD_EXPERIMENT_CODE_PATH=path_store.experiment_code,
    )
    return experiment_config


def extract_dataset_name(experiment_name: str) -> str:
    experiment_config = load_experiment_config(experiment_name)
    runner_type = experiment_config.pop("type")
    runner_config = experiment_config.pop("config")
    run_module = importlib.import_module(f"appworld_agents.code.{runner_type}.run")
    dataset_name = run_module.extract_dataset_name(runner_config)
    return cast(str, dataset_name)


def extract_runner_config(experiment_name: str) -> dict[str, Any]:
    experiment_config = load_experiment_config(experiment_name)
    runner_config = experiment_config.pop("config")
    return cast(dict[str, Any], runner_config)


def extract_metadata(experiment_name: str) -> dict[str, Any]:
    experiment_config = load_experiment_config(experiment_name)
    metadata = experiment_config.get("metadata", {})
    return cast(dict[str, Any], metadata)


@app.command()
def install(
    repo: bool = typer.Option(
        False,
        help=(
            "1. --no-repo (default) unpacks encrypted code in the AppWorld package. "
            "Run it anywhere. "
            "2. --repo unpacks encrypted code in the AppWorld repository. "
            "Run it from the repository root. "
        ),
    ),
) -> None:
    """
    [   Setup   ] Unpack encrypted portion of the AppWorld code.
    """
    from appworld.install import install_package, install_repo

    if repo:
        install_repo()
    else:
        install_package()


download_app = typer.Typer(no_args_is_help=True, cls=DownloadOrderCommands)


@download_app.command("data")
def download_data(
    mode: DataDownloadMode = typer.Option(
        DataDownloadMode.minimal,
        help=(
            "The mode in which to download the ground-truths for the test task. "
            "The train and dev tasks always come with 'full' ground-truths. "
            "The 'partial' and 'full' modes for test tasks require private keys "
            "which only us AppWorld maintainers have right now."
        ),
    ),
    version: str | None = typer.Option(
        None,
        help=(
            "The version of the data to download. "
            "If not specified, the latest version will be downloaded."
        ),
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory, where the data or experiment outputs "
            "should be downloaded. Pass only if it is not the current directory."
        ),
    ),
) -> None:
    """
    [   Setup   ] Download AppWorld data
    """
    from appworld.download import download_data

    update_root(root)
    verify_fully_installed(with_setup=with_setup)
    download_data(version, mode=mode.value)


@download_app.command("experiment-outputs")
def download_experiment_outputs(
    version: str | None = typer.Option(
        None,
        help=(
            "The version of the experiment outputs to download. "
            "If not specified, the latest version will be downloaded."
        ),
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory, where the data or experiment outputs "
            "should be downloaded. Pass only if it is not the current directory."
        ),
    ),
) -> None:
    """
    [   Setup   ] Download AppWorld agents' experiment outputs.
    """
    from appworld.download import download_experiment_outputs

    update_root(root)
    verify_fully_installed(with_setup=with_setup)
    download_experiment_outputs(version)


app.add_typer(
    download_app,
    name="download",
    help="[   Setup   ] Download AppWorld data or baseline agents' experiment outputs.",
)


verify_app = typer.Typer(no_args_is_help=True, cls=VerifyOrderCommands)


def _parse_json_string(variable_name: str, variable_value: str | None) -> dict[str, Any] | None:
    if variable_value is None:
        return None
    try:
        parsed_value = json.loads(variable_value)
        if not isinstance(parsed_value, dict):
            raise ValueError
        return parsed_value
    except (json.JSONDecodeError, ValueError) as exception:
        raise ValueError(
            f"The {variable_name} must be a valid JSON string representing a dictionary. "
            f"Got: {variable_value}"
        ) from exception


@verify_app.command("tests")
def verify_tests(
    verbose: bool = typer.Option(
        True,
        help="Whether to run the tests with verbose logs.",
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [   Setup   ] Verify AppWorld installation by running unit tests.
    """
    from appworld.verify import verify_tests

    update_root(root)
    verify_data_exists(with_setup=with_setup)
    verify_fully_installed(with_setup=with_setup)
    success = verify_tests(verbose=verbose)
    if not success:
        # NOTE: this needs to raise an error for the CI to catch it as a failure.
        raise Exception("AppWorld tests verification failed.")


@verify_app.command("tasks")
def verify_tasks(
    remote_apis_url: str | None = typer.Option(
        None,
        help="The URL of the remote APIs server to use for task verification.",
    ),
    remote_mcp_url: str | None = typer.Option(
        None,
        help=(
            "The URL of the remote MCP server to use for task verification. "
            "Set it to `stdio` for stdio transport and `http` for HTTP transport. "
            "If passed, the remote_apis_url must also be passed."
        ),
    ),
    remote_environment_url: str | None = typer.Option(
        None,
        help="The URL of the remote environment server to use for task verification.",
    ),
    remote_docker: bool = typer.Option(
        False,
        help="Whether any of the remote servers are running in a Docker container.",
    ),
    apis_server_kwargs: str | None = typer.Option(
        None,
        help="Additional arguments for the APIs server as a JSON string.",
    ),
    environment_server_kwargs: str | None = typer.Option(
        None,
        help="Additional arguments for the environment server as a JSON string.",
    ),
    mcp_server_kwargs: str | None = typer.Option(
        None,
        help="Additional arguments for the MCP server as a JSON string.",
    ),
    docker_tag: str = typer.Option(
        "latest",
        help="The Docker image tag to use if remote_docker flag is enabled.",
    ),
    include_test_tasks: bool = typer.Option(
        False,
        help="Whether to include test tasks in the verification (for internal use only).",
    ),
    include_only_first_n_tasks: int = typer.Option(
        -1, help="The number of tasks to include in the verification. -1 means all tasks."
    ),
    num_processes: int = typer.Option(
        1, help="The number of parallel processes to use for task verification."
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [   Setup   ] Verify AppWorld installation by running end-to-end task verifications.
    """
    from appworld.verify import verify_tasks

    update_root(root)
    verify_data_exists(with_setup=with_setup)
    verify_fully_installed(with_setup=with_setup)
    parsed_apis_server_kwargs = _parse_json_string("apis_server_kwargs", apis_server_kwargs)
    parsed_environment_server_kwargs = _parse_json_string(
        "environment_server_kwargs", environment_server_kwargs
    )
    parsed_mcp_server_kwargs = _parse_json_string("mcp_server_kwargs", mcp_server_kwargs)
    success = verify_tasks(
        remote_apis_url=remote_apis_url,
        remote_mcp_url=remote_mcp_url,
        remote_environment_url=remote_environment_url,
        remote_docker=remote_docker,
        apis_server_kwargs=parsed_apis_server_kwargs,
        environment_server_kwargs=parsed_environment_server_kwargs,
        mcp_server_kwargs=parsed_mcp_server_kwargs,
        docker_tag=docker_tag,
        include_test_tasks=include_test_tasks,
        include_only_first_n_tasks=include_only_first_n_tasks,
        num_processes=num_processes,
    )
    if not success:
        # NOTE: this needs to raise an error for the CI to catch it as a failure.
        raise Exception("AppWorld tasks verification failed.")


app.add_typer(
    verify_app,
    name="verify",
    help="[   Setup   ] Verify AppWorld installation.",
)


@app.command()
def explore(
    dataset_name: ExtendedDatasetName | None = typer.Option(
        None,
        help="The dataset split the task belongs to.",
    ),
    difficulty: Difficulty | None = typer.Option(
        None,
        help="The difficulty level of the task (1: easy, 2: medium, 3: hard).",
    ),
    num_tasks: int = typer.Option(
        50,
        help="The maximum number of tasks to display.",
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Development] Explore task instructions in the AppWorld dataset.
    """
    from appworld.task import Task, load_task_ids, valid_dataset_names

    update_root(root)
    verify_data_exists(with_setup=with_setup)
    task_ids: list[str] = []
    task_id_to_dataset_name = {}
    difficulty_value = difficulty.value if difficulty else None
    for datset_name_ in valid_dataset_names():
        if not dataset_name or datset_name_ == dataset_name:
            task_ids_ = load_task_ids(dataset_name=datset_name_, difficulty=difficulty_value)
            for task_id_ in task_ids_:
                task_id_to_dataset_name[task_id_] = datset_name_
            task_ids.extend(task_ids_)
    random.shuffle(task_ids)
    task_ids = task_ids[:num_tasks]
    if not task_ids:
        print("[yellow]No tasks match your filters.[/yellow]")
        return
    table = Table(title="Filtered task instructions")
    table.add_column("#", style="cyan")
    table.add_column("ID", style="cyan")
    table.add_column("Set", style="green")
    table.add_column("Difficulty", style="blue")
    table.add_column("Description", style="white")
    for index, task_id in enumerate(task_ids):
        task = Task.load(task_id, load_ground_truth=True)
        assert hasattr(task, "ground_truth") and task.ground_truth is not None  # mypy
        difficulty_int_ = str(task.ground_truth.metadata["difficulty"])
        difficulty_str_ = {"1": "easy (1)", "2": "medium (2)", "3": "hard (3)"}[difficulty_int_]
        table.add_row(
            str(index + 1),
            str(task_id),
            task_id_to_dataset_name[task_id],
            difficulty_str_,
            task.instruction,
        )
        table.add_row("", "", "", "", "")
    rprint(table)


serve_app = typer.Typer(no_args_is_help=True, cls=ServeOrderCommands)


def _serve_inner_wrapper(
    _serve_inner: Callable[..., None], queue: Queue[BaseException], kwargs: dict[str, Any]
) -> None:
    try:
        _serve_inner(num_servers=1, **kwargs)
    except BaseException as e:
        queue.put(e)


def _launch_servers(
    _serve_inner: Callable[..., None], list_of_kwargs: list[dict[str, Any]]
) -> None:
    queue: Queue[BaseException] = Queue()
    processes = [
        Process(target=_serve_inner_wrapper, args=(_serve_inner, queue, kwargs))
        for kwargs in list_of_kwargs
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if not queue.empty():
        raise queue.get()


def _local_url(port: int) -> str:
    return f"http://localhost:{port}"


def _show_server_urls(dict_of_server_urls: dict[str, Any]) -> None:
    print_rule("AppWorld Server URLs")
    print(json.dumps(dict_of_server_urls, indent=4))
    print_rule()
    print("\n\n")


def _serve_multiple_inner(
    environment: str | None = None,
    apis: str | None = None,
    mcp: str | None = None,
    uis: str | None = None,
    num_servers: int = 1,
    show_server_urls: bool = True,
) -> None:
    if num_servers > 1:
        if not all("--port -1" in e for e in [environment, apis, mcp, uis] if e is not None):
            raise ValueError("When num_servers > 1, all individual server ports must be -1.")
        list_of_kwargs: list[dict[str, Any]] = []
        dict_of_server_urls: dict[str, list[str]] = defaultdict(list)
        for _ in range(num_servers):
            environment_ = environment
            if environment is not None:
                port = get_available_port()
                environment_ = environment.replace("--port -1", f"--port {port}")
                dict_of_server_urls["remote_environment_url"].append(_local_url(port))
            apis_ = apis
            if apis is not None:
                port = get_available_port()
                apis_ = apis.replace("--port -1", f"--port {port}")
                dict_of_server_urls["remote_apis_url"].append(_local_url(port))
            mcp_ = mcp
            if mcp is not None:
                port = get_available_port()
                mcp_ = mcp.replace("--port -1", f"--port {port}")
                dict_of_server_urls["remote_mcp_url"].append(_local_url(port))
            uis_ = uis
            if uis is not None:
                port = get_available_port()
                uis_ = uis.replace("--port -1", f"--port {port}")
                dict_of_server_urls["remote_uis_url"].append(_local_url(port))
            list_of_kwargs.append(
                {
                    "environment": environment_,
                    "apis": apis_,
                    "mcp": mcp_,
                    "uis": uis_,
                }
            )
        for kwargs in list_of_kwargs:
            kwargs["show_server_urls"] = False
        _show_server_urls(dict_of_server_urls)
        _launch_servers(
            _serve_multiple_inner,
            list_of_kwargs=list_of_kwargs,
        )
    kwargs = {"environment": environment, "apis": apis, "mcp": mcp, "uis": uis}
    for key, value in kwargs.items():
        if value is not None:
            for name in ["--root", "--docker"]:
                if name in value:
                    raise ValueError(
                        f"The {name} option should not be passed in individual server CLI args, but outside of it. "
                        f"For '{key}' server, got CLI args: '{value}'"
                    )
            if "--show-usage" in value or "--no-show-usage" in value:
                raise ValueError(
                    "The --show-usage or --no-show-usage option should not be passed in individual server CLI args. "
                    f"For '{key}' server, got CLI args: '{value}'"
                )

    dict_of_server_urls = {}
    if environment is not None:
        port = DEFAULT_REMOTE_ENVIRONMENT_PORT
        match = re.match(r".*--port +(-?\d+).*", environment)
        if match:
            port = int(match.group(1))
            if port == -1:
                port = get_available_port()
                environment = re.sub(r"--port +(-1)", f"--port {port}", environment)
        dict_of_server_urls["remote_environment_url"] = [_local_url(port)]
    if apis is not None:
        port = DEFAULT_REMOTE_APIS_PORT
        match = re.match(r".*--port +(-?\d+).*", apis)
        if match:
            port = int(match.group(1))
            if port == -1:
                port = get_available_port()
                apis = re.sub(r"--port +(-1)", f"--port {port}", apis)
        dict_of_server_urls["remote_apis_url"] = [_local_url(port)]
    if mcp is not None:
        port = DEFAULT_REMOTE_MCP_PORT
        match = re.match(r".*--port +(-?\d+).*", mcp)
        if match:
            port = int(match.group(1))
            if port == -1:
                port = get_available_port()
                mcp = re.sub(r"--port +(-1)", f"--port {port}", mcp)
        dict_of_server_urls["remote_mcp_url"] = [_local_url(port)]
    if uis is not None:
        port = DEFAULT_REMOTE_UIS_PORT
        match = re.match(r".*--port +(-?\d+).*", uis)
        if match:
            port = int(match.group(1))
            if port == -1:
                port = get_available_port()
                uis = re.sub(r"--port +(-1)", f"--port {port}", uis)
        dict_of_server_urls["remote_uis_url"] = [_local_url(port)]
    if show_server_urls:
        _show_server_urls(dict_of_server_urls)

    subprocesses: list[subprocess.Popen[Any]] = []

    def spawn(command: str) -> subprocess.Popen[Any]:
        print(f"Running command: {command}")
        if "--docker" in command:
            raise Exception(
                "The --docker option should not be passed in individual server args, but outside of it"
            )
        env = os.environ
        env["NO_SHOW_URLS"] = "1"
        args = shlex.split(command, posix=(sys.platform != "win32"))
        process = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr, env=env)
        subprocesses.append(process)
        return process

    def cleanup(exit_code: int = 0, *_: int | None, **__: dict[str, Any]) -> None:
        for process in subprocesses:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        sys.exit(exit_code)

    signal.signal(signal.SIGINT, lambda *_: cleanup(0))
    signal.signal(signal.SIGTERM, lambda *_: cleanup(0))

    def maybe_unwrap_str(cmd: str) -> str:
        cmd = cmd.strip()
        if cmd.startswith('"') and cmd.endswith('"'):
            return cmd[1:-1].strip()
        if cmd.startswith("'") and cmd.endswith("'"):
            return cmd[1:-1].strip()
        return cmd

    if environment is not None:
        command = f"{sys.executable} -m appworld.cli serve environment --no-show-usage {maybe_unwrap_str(environment)} --root {path_store.root}"
        spawn(command)
    if apis is not None:
        command = f"{sys.executable} -m appworld.cli serve apis --no-show-usage {maybe_unwrap_str(apis)} --root {path_store.root}"
        spawn(command)
    if mcp is not None:
        if "--remote-apis-url" not in mcp and "remote_apis_url" not in dict_of_server_urls:
            raise ValueError("When serving MCP, the remote_apis_url must available")
        if "--remote-apis-url" not in mcp:
            remote_apis_url = dict_of_server_urls["remote_apis_url"][0]
            mcp = f"{maybe_unwrap_str(mcp)} --remote-apis-url {maybe_unwrap_str(remote_apis_url)}"
        command = f"{sys.executable} -m appworld.cli serve mcp {maybe_unwrap_str(mcp)} --root {path_store.root}"
        spawn(command)
    if uis is not None:
        if "--remote-apis-url" not in uis and "remote_apis_url" not in dict_of_server_urls:
            raise ValueError("When serving UIs, the remote_apis_url must be available")
        if "--remote-apis-url" not in uis:
            remote_apis_url = dict_of_server_urls["remote_apis_url"][0]
            uis = f"{maybe_unwrap_str(uis)} --remote-apis-url {maybe_unwrap_str(remote_apis_url)}"
        command = f"{sys.executable} -m appworld.cli serve uis --no-show-usage {maybe_unwrap_str(uis)} --root {path_store.root}"
        spawn(command)
    # Wait for all subprocesses to complete and fail if any of them fails
    while subprocesses:
        for process in list(subprocesses):
            ret = process.poll()
            if ret is not None:  # means process has finished
                subprocesses.remove(process)
                if ret != 0:
                    print(f"The server '{process.args!s}' exited with code {ret}. Stopping all.")
                    cleanup(ret)
        try:
            subprocesses[0].wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _serve_environment_inner(
    port: int, num_servers: int = 1, show_server_urls: bool = True
) -> None:
    if num_servers > 1:
        if port != -1:
            raise ValueError("When num_servers > 1, port must be -1 to auto-assign ports.")
        ports = [get_available_port() for _ in range(num_servers)]
        list_of_kwargs = [{"port": port} for port in ports]
        dict_of_server_urls = {"remote_environment_url": [_local_url(port) for port in ports]}
        for kwargs in list_of_kwargs:
            kwargs["show_server_urls"] = False
        _show_server_urls(dict_of_server_urls)
        _launch_servers(
            _serve_environment_inner,
            list_of_kwargs=list_of_kwargs,
        )
    if port == -1:
        port = get_available_port()
    if show_server_urls and os.environ.get("NO_SHOW_URLS", "0") != "1":
        dict_of_server_urls = {"remote_environment_url": [_local_url(port)]}
        _show_server_urls(dict_of_server_urls)
    importlib.import_module("appworld.serve.environment").run(port=port)


def _serve_apis_inner(
    port: int,
    num_workers: int = 1,
    on_disk: bool = False,
    num_servers: int = 1,
    show_server_urls: bool = True,
) -> None:
    if num_servers > 1:
        if port != -1:
            raise ValueError("When num_servers > 1, port must be -1 to auto-assign ports.")
        ports = [get_available_port() for _ in range(num_servers)]
        list_of_kwargs = [{"port": port} for port in ports]
        dict_of_server_urls = {"remote_apis_url": [_local_url(port) for port in ports]}
        _show_server_urls(dict_of_server_urls)
        for kwargs in list_of_kwargs:
            kwargs["show_server_urls"] = False
        _launch_servers(
            _serve_apis_inner,
            list_of_kwargs=list_of_kwargs,
        )
    if port == -1:
        port = get_available_port()
    if show_server_urls and os.environ.get("NO_SHOW_URLS", "0") != "1":
        dict_of_server_urls = {"remote_apis_url": [_local_url(port)]}
        _show_server_urls(dict_of_server_urls)
    importlib.import_module("appworld.serve.apis").run(
        port=port, num_workers=num_workers, on_disk=on_disk
    )


def _serve_mcp_inner(
    port: int,
    transport: Transport = Transport.stdio,
    output_type: MCPOutputType = MCPOutputType.structured_data_only,
    app_names: list[str] | None = None,
    remote_apis_url: str | None = None,
    num_servers: int = 1,
    show_server_urls: bool = False,  # Need to be False as it may interfere with MCP communication over stdio
) -> None:
    remote_apis_url = remote_apis_url or DEFAULT_REMOTE_APIS_URL
    if num_servers > 1:
        if port != -1:
            raise ValueError("When num_servers > 1, port must be -1 to auto-assign ports.")
        ports = [get_available_port() for _ in range(num_servers)]
        list_of_kwargs = [
            {
                "port": port,
                "transport": transport,
                "output_type": output_type,
                "app_names": app_names,
                "remote_apis_url": remote_apis_url,
            }
            for port in ports
        ]
        dict_of_server_urls = {
            "remote_mcp_url": [_local_url(port) for port in ports],
            "remote_apis_url": [remote_apis_url]
            * len(ports),  # TODO: Needs to be spawned individually.
        }
        for kwargs in list_of_kwargs:
            kwargs["show_server_urls"] = False
        _show_server_urls(dict_of_server_urls)
        _launch_servers(_serve_mcp_inner, list_of_kwargs=list_of_kwargs)
    ensure_package_installed("mcp")
    if port == -1:
        port = get_available_port()
    if show_server_urls and os.environ.get("NO_SHOW_URLS", "0") != "1":
        dict_of_server_urls = {"remote_mcp_url": [_local_url(port)]}
        _show_server_urls(dict_of_server_urls)
    importlib.import_module("appworld.serve._mcp").run(
        transport=transport.value,
        app_names=app_names,
        output_type=output_type,
        remote_apis_url=remote_apis_url,
        port=port,
    )


def _serve_uis_inner(
    port: int,
    num_workers: int = 1,
    remote_apis_url: str | None = None,
    num_servers: int = 1,
    show_server_urls: bool = True,
) -> None:
    remote_apis_url = remote_apis_url or DEFAULT_REMOTE_APIS_URL
    if num_servers > 1:
        if port != -1:
            raise ValueError("When num_servers > 1, port must be -1 to auto-assign ports.")
        ports = [get_available_port() for _ in range(num_servers)]
        list_of_kwargs = [
            {
                "port": port,
                "num_workers": num_workers,
                "remote_apis_url": remote_apis_url,
            }
            for port in ports
        ]
        dict_of_server_urls = {
            "remote_uis_url": [_local_url(port) for port in ports],
            "remote_apis_url": [remote_apis_url]
            * len(ports),  # TODO: Needs to be spawned individually.
        }
        for kwargs in list_of_kwargs:
            kwargs["show_server_urls"] = False
        _show_server_urls(dict_of_server_urls)
        _launch_servers(_serve_uis_inner, list_of_kwargs=list_of_kwargs)
    if port == -1:
        port = get_available_port()
    if show_server_urls and os.environ.get("NO_SHOW_URLS", "0") != "1":
        dict_of_server_urls = {"remote_uis_url": [_local_url(port)]}
        _show_server_urls(dict_of_server_urls)
    importlib.import_module("appworld.serve.uis").run(
        remote_apis_url=remote_apis_url,
        port=port,
        num_workers=num_workers,
    )


def _serve_outer(
    server_type: ServerType,
    port: int | None = None,
    show_usage: bool = True,
    docker: bool = False,
    docker_tag: str = "latest",
    num_servers: int = 1,
    with_setup: bool = False,
    root: str | None = None,
    **kwargs: Any,
) -> None:
    update_root(root)
    verify_data_exists(with_setup=with_setup)
    verify_fully_installed(with_setup=with_setup)
    server_type_str = {
        ServerType.multiple: "multiple",
        ServerType.environment: "environment",
        ServerType.apis: "apis",
        ServerType.mcp: "mcp",
        ServerType.uis: "uis",
    }[server_type]
    if num_servers > 1 and isinstance(port, int) and port != -1:
        raise ValueError("When num_servers > 1, port must be -1 to auto-assign ports.")
    if num_servers > 1 and port is None and server_type != ServerType.multiple:
        show_usage = False
        port = -1
    kwargs["num_servers"] = num_servers
    if port is None and server_type != ServerType.multiple:
        port = {
            ServerType.environment: DEFAULT_REMOTE_ENVIRONMENT_PORT,
            ServerType.apis: DEFAULT_REMOTE_APIS_PORT,
            ServerType.uis: DEFAULT_REMOTE_UIS_PORT,
        }[server_type]
    if show_usage:
        help_function = {
            ServerType.multiple: show_serve_multiple_help,
            ServerType.environment: show_serve_environment_help,
            ServerType.apis: show_serve_apis_help,
            ServerType.mcp: show_serve_mcp_help,
            ServerType.uis: show_serve_uis_help,
        }[server_type]
        if port is not None:
            help_function(port, docker)
    if not docker:
        serve_inner_ = {
            ServerType.multiple: _serve_multiple_inner,
            ServerType.environment: _serve_environment_inner,
            ServerType.apis: _serve_apis_inner,
            ServerType.mcp: _serve_mcp_inner,
            ServerType.uis: _serve_uis_inner,
        }[server_type]
        serve_inner: Callable[..., None] = cast(Callable[..., None], serve_inner_)
        if port is not None:
            kwargs["port"] = port
        if ServerType.mcp == server_type:
            kwargs["show_server_urls"] = False  # It interferes with MCP communication
        serve_inner(**kwargs)
    else:
        image_name = f"ghcr.io/stonybrooknlp/appworld:{docker_tag}"
        root_directory = os.path.abspath(path_store.root)
        env_args = ["-e", f"APPWORLD_ROOT={root_directory}"]
        volume_args = [
            "-v",
            f"{root_directory}/data:/run/data:rw",
            "-v",
            f"{root_directory}/experiments/outputs:/run/experiments/outputs:rw",
        ]
        port_args = []
        if port is not None:
            port_args.extend(["-p", f"{port}:{port}"])
        server_type_to_default_port = {
            "apis": DEFAULT_REMOTE_APIS_PORT,
            "environment": DEFAULT_REMOTE_ENVIRONMENT_PORT,
            "mcp": DEFAULT_REMOTE_MCP_PORT,
            "uis": DEFAULT_REMOTE_UIS_PORT,
        }
        for key, value in kwargs.items():
            for api_type, default_port in server_type_to_default_port.items():
                if key == api_type and value is not None:
                    value = value.strip() or "{}"
                    match = re.search(r"--port +(\d+)", value)
                    port_ = default_port
                    if match:
                        port_ = int(match.group(1))
                    port_args.extend(["-p", f"{port_}:{port_}"])
        main_args = [image_name, server_type_str]
        if port is not None:
            main_args.extend(["--port", str(port)])
        if server_type != ServerType.multiple:
            main_args.extend(["--no-show-usage"])
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if kwargs:
            main_args.extend(
                shlex.split(build_cli_command("", kwargs), posix=(sys.platform != "win32"))
            )
        command = ["docker", "run", *env_args, *volume_args, *port_args, *main_args]
        print()
        print_rule("Docker Command (Run manually if it fails)")
        command_str = " ".join(command)
        print(command_str)
        print_rule()
        print()
        subprocess.run(command)


@serve_app.command("environment")
def serve_environment(
    port: int | None = typer.Option(
        None,
        help="The port number to serve on.",
    ),
    show_usage: bool | None = typer.Option(
        True,
        help="Show usage notes before starting the server.",
    ),
    docker: bool = typer.Option(
        False,
        help=(
            "Whether to run it in a Docker container. "
            "The docker will have access to only ./data and ./experiments/outputs "
            "directories relative to the AppWorld root directory."
        ),
    ),
    docker_tag: str = typer.Option(
        "latest",
        help="The Docker image tag to use if docker flag is enabled.",
    ),
    num_servers: int = typer.Option(
        1,
        min=1,
        help=(
            "The number of servers to launch. "
            "Port must be -1 to auto-assign ports when launching multiple servers."
        ),
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Development] Serve AppWorld environment with or without Docker.
    """
    show_usage = True if show_usage is None else show_usage
    _serve_outer(
        server_type=ServerType.environment,
        port=port,
        show_usage=show_usage,
        docker=docker,
        docker_tag=docker_tag,
        num_servers=num_servers,
        with_setup=with_setup,
        root=root,
    )


@serve_app.command("apis")
def serve_apis(
    port: int | None = typer.Option(
        None,
        help="The port number to serve on.",
    ),
    num_workers: int = typer.Option(
        1,
        min=1,
        help="The number of worker processes to use for serving.",
    ),
    on_disk: bool = typer.Option(
        False,
        help="Whether to use on-disk DB storage instead of in-memory.",
    ),
    show_usage: bool | None = typer.Option(
        True,
        help="Show usage notes before starting the server.",
    ),
    docker: bool = typer.Option(
        False,
        help=(
            "Whether to run it in a Docker container. "
            "The docker will have access to only ./data and ./experiments/outputs "
            "directories relative to the AppWorld root directory."
        ),
    ),
    docker_tag: str = typer.Option(
        "latest",
        help="The Docker image tag to use if docker flag is enabled.",
    ),
    num_servers: int = typer.Option(
        1,
        min=1,
        help=(
            "The number of servers to launch. "
            "Port must be -1 to auto-assign ports when launching multiple servers."
        ),
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Development] Serve AppWorld APIs with or without Docker.
    """
    show_usage = True if show_usage is None else show_usage
    _serve_outer(
        server_type=ServerType.apis,
        port=port,
        show_usage=show_usage,
        docker=docker,
        docker_tag=docker_tag,
        num_servers=num_servers,
        with_setup=with_setup,
        root=root,
        num_workers=num_workers,
        on_disk=on_disk,
    )


@serve_app.command("mcp")
def serve_mcp(
    transport: Transport = typer.Argument(
        ...,
        help="The transport protocol to use.",
    ),
    port: int | None = typer.Option(
        DEFAULT_REMOTE_MCP_PORT,
        help="The port number to serve on (if using HTTP transport).",
    ),
    output_type: MCPOutputType | None = typer.Option(
        MCPOutputType.structured_data_only,
        help=(
            "The type of output to the call_tool from the MCP server should return: "
            "'content_only' returns only the content, 'structured_data_only' returns structured data, "
            "and 'both' returns both in a tuple."
        ),
    ),
    app_names: str | None = typer.Option(
        None,
        help="A comma-separated list of app names to serve. If not provided, defaults to all apps.",
    ),
    remote_apis_url: str = typer.Option(
        DEFAULT_REMOTE_APIS_URL, help="The URL where the AppWorld APIs server is running"
    ),
    docker: bool = typer.Option(
        False,
        help=(
            "Whether to run it in a Docker container. "
            "The docker will have access to only ./data and ./experiments/outputs "
            "directories relative to the AppWorld root directory."
        ),
    ),
    docker_tag: str = typer.Option(
        "latest",
        help="The Docker image tag to use if docker flag is enabled.",
    ),
    num_servers: int = typer.Option(
        1,
        min=1,
        help=(
            "The number of servers to launch. "
            "Port must be -1 to auto-assign ports when launching multiple servers."
        ),
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Development] Serve AppWorld MCP.
    """
    app_names_: list[str] | None = None
    if app_names is not None:
        app_names_ = [name.strip() for name in app_names.split(",") if name.strip()]
    _serve_outer(
        server_type=ServerType.mcp,
        port=port,
        show_usage=False,  # It interferes with stdio transport
        docker=docker,
        docker_tag=docker_tag,
        num_servers=num_servers,
        with_setup=with_setup,
        root=root,
        transport=transport,
        output_type=output_type,
        app_names=app_names_,
        remote_apis_url=remote_apis_url,
    )


# @serve_app.command("uis")  # TODO: Uncomment with the next release
def serve_uis(
    port: int | None = typer.Option(
        None,
        help="The port number to serve on.",
    ),
    num_workers: int = typer.Option(
        1,
        min=1,
        help="The number of worker processes to use for serving.",
    ),
    show_usage: bool | None = typer.Option(
        True,
        help="Show usage notes before starting the server.",
    ),
    docker: bool = typer.Option(
        False,
        help=(
            "Whether to run it in a Docker container. "
            "The docker will have access to only ./data and ./experiments/outputs "
            "directories relative to the AppWorld root directory."
        ),
    ),
    docker_tag: str = typer.Option(
        "latest",
        help="The Docker image tag to use if docker flag is enabled.",
    ),
    num_servers: int = typer.Option(
        1,
        min=1,
        help=(
            "The number of servers to launch. "
            "Port must be -1 to auto-assign ports when launching multiple servers."
        ),
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Development] Serve AppWorld UIs with or without Docker.
    """
    show_usage = True if show_usage is None else show_usage
    _serve_outer(
        server_type=ServerType.uis,
        port=port,
        show_usage=show_usage,
        docker=docker,
        docker_tag=docker_tag,
        num_servers=num_servers,
        with_setup=with_setup,
        root=root,
        num_workers=num_workers,
    )


@serve_app.command("multiple")
def serve_multiple(
    environment: str | None = typer.Option(
        None,
        help=(
            "Environment server CLI arguments as a string. "
            "This is what you would typically pass to 'appworld serve environment' command. "
            f"E.g., --environment '--port {DEFAULT_REMOTE_ENVIRONMENT_PORT}'. "
            "To go with defaults, pass an empty string ''."
        ),
    ),
    apis: str | None = typer.Option(
        None,
        help=(
            "APIs server CLI arguments as a string. "
            "This is what you would typically pass to 'appworld serve apis' command. "
            f"E.g., --apis '--port {DEFAULT_REMOTE_APIS_PORT}'. "
            "To go with defaults, pass an empty string ''."
        ),
    ),
    mcp: str | None = typer.Option(
        None,
        help=(
            "MCP server CLI arguments as a string. "
            "This is what you would typically pass to 'appworld serve mcp' command. "
            f"E.g., --mcp '--port {DEFAULT_REMOTE_MCP_PORT}'. "
            "To go with defaults, pass an empty string ''."
        ),
    ),
    # uis: str | None = typer.Option(  # TODO: Uncomment with the next release
    #     None,
    #     help=(
    #         "UIs server CLI arguments as a string. "
    #         "This is what you would typically pass to 'appworld serve uis' command. "
    #         f"E.g., --uis '--port {DEFAULT_REMOTE_UIS_PORT}'. "
    #         "To go with defaults, pass an empty string ''."
    #     ),
    # ),
    docker: bool = typer.Option(
        False,
        help=(
            "Whether to run it in a Docker container. "
            "The docker will have access to only ./data and ./experiments/outputs "
            "directories relative to the AppWorld root directory. "
            "This argument should not be passed in individual server CLI args."
        ),
    ),
    docker_tag: str = typer.Option(
        "latest",
        help="The Docker image tag to use if docker flag is enabled.",
    ),
    num_servers: int = typer.Option(
        1,
        min=1,
        help=(
            "The number of servers to launch. "
            "Port must be -1 to auto-assign ports when launching multiple servers."
        ),
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory. "
            "This argument should not be passed in individual server CLI args."
        ),
    ),
) -> None:
    """
    [Development] Serve multiple AppWorld servers together.
    """
    # if not non_none([environment, apis, mcp, uis]):  # TODO: Uncomment with the next release
    if not non_none([environment, apis, mcp]):
        print("Error: At least one server's CLI arguments must be provided to run.")
        return
    _serve_outer(
        server_type=ServerType.multiple,
        port=None,
        show_usage=False,
        docker=docker,
        docker_tag=docker_tag,
        num_servers=num_servers,
        with_setup=with_setup,
        root=root,
        environment=environment,
        apis=apis,
        mcp=mcp,
        # uis=uis,  # TODO: Uncomment with the next release
    )


app.add_typer(
    serve_app,
    name="serve",
    help="[Development] Serve AppWorld Environment, APIs, MCP or UIs with or without Docker.",
)


@app.command()
def play(
    port: int | None = typer.Option(
        DEFAULT_REMOTE_ENVIRONMENT_PORT,
        help="The port number to serve playground/demo on.",
    ),
    docker: bool = typer.Option(
        False,
        help=(
            "Whether to run it in a Docker container. "
            "The docker will have access to only ./data and ./experiments/outputs "
            "directories relative to the AppWorld root directory."
        ),
    ),
    docker_tag: str = typer.Option(
        "latest",
        help="The Docker image tag to use if docker flag is enabled.",
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Development] Start an interactive coding playground to explore task worlds.
    """
    print_text("🚀 Starting AppWorld Playground!", style="bold green")
    print_text(
        f"\n👉 Visit: http://0.0.0.0:{port}/playground after the server starts.\n",
        style="bold cyan",
    )
    _serve_outer(
        server_type=ServerType.environment,
        port=port,
        show_usage=False,
        docker=docker,
        docker_tag=docker_tag,
        with_setup=with_setup,
        root=root,
    )


def get_run_options(
    config_directory: str,
    use_color: bool = True,
    include_path_in_label: bool = True,
) -> tuple[str, Tree | None]:
    def build_tree(directory_root: str, jsonnet_files: list[str], root_label_text: str) -> Tree:
        tree_structure = Tree(f"[bold magenta]{root_label_text}[/bold magenta]")
        directory_mapping: dict[str, Any] = {}
        for file_path in jsonnet_files:
            relative_path = os.path.relpath(file_path, directory_root)
            path_components = relative_path.split(os.sep)
            current_level: dict[str, Any] = directory_mapping
            for folder in path_components[:-1]:
                current_level = cast(dict[str, Any], current_level.setdefault(folder, {}))
            files_list = cast(list[str], current_level.setdefault("__files__", []))
            files_list.append(relative_path)

        def add_branches_to_tree(current_tree: Tree, current_node: dict[str, Any]) -> None:
            for node_key in sorted(current_node.keys()):
                if node_key == "__files__":
                    for relative_file_path in sorted(current_node["__files__"]):
                        file_stem = os.path.splitext(os.path.basename(relative_file_path))[0]
                        full_experiment_path = os.path.splitext(relative_file_path)[0]
                        current_tree.add(
                            f"[green]{file_stem}[/green] ([dim]{full_experiment_path}[/dim])"
                        )
                else:
                    branch = current_tree.add(f"[cyan]{node_key}[/cyan]")
                    add_branches_to_tree(branch, current_node[node_key])

        add_branches_to_tree(tree_structure, directory_mapping)
        return tree_structure

    all_jsonnet_files = []
    for root_directory, _, file_list in os.walk(config_directory):
        for file_name in file_list:
            if file_name.endswith(".jsonnet"):
                all_jsonnet_files.append(os.path.join(root_directory, file_name))
    if not all_jsonnet_files:
        output_message = "No .jsonnet config files found!"
        if use_color:
            output_message = "[red]No .jsonnet config files found![/red]"
        return output_message, None
    tree_root_label = "# Experiment configs"
    if include_path_in_label:
        tree_root_label += f" found in {config_directory}"
    tree = build_tree(config_directory, all_jsonnet_files, tree_root_label)
    console_output_buffer = StringIO()
    console = Console(
        file=console_output_buffer,
        force_terminal=True,
        color_system="truecolor" if use_color else None,
        no_color=not use_color,
    )
    console.print(tree)
    return console_output_buffer.getvalue(), tree


@app.command()
def run(
    experiment_name: str = typer.Argument(
        ...,
        help=(
            "Name of the experiment to run. E.g., {experiment_name} corresponds "
            "to the config in ./configs/{experiment_name}.jsonnet "
            "relative to where appworld-agents is installed. "
            "Set to 'options' to see all experiment name options. Set to 'auto' to generate "
            "config automatically based on model, agent and dataset names, before running the experiment."
        ),
    ),
    model_name: str | None = typer.Option(
        None,
        help=(
            "Name of the model to run. Set to 'options' to see all model name options. "
            "Applicable only if experiment_name is set to 'auto'."
        ),
    ),
    agent_name: str | None = typer.Option(
        None,
        help=(
            "Name of the agent to run. Set to 'options' to see all agent name options. "
            "Applicable only if experiment_name is set to 'auto'."
        ),
    ),
    dataset_name: str | None = typer.Option(
        None,
        help=(
            "Name of the dataset to run on. Set to 'options' to see all dataset name options. "
            "Applicable only if experiment_name is set to 'auto'."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        help=(
            "If experiment_name is set to 'auto' and dry_run is true, "
            "the experiment config will be generated and shown but not run. "
            "This parameter is not applicable if experiment_name is not 'auto'."
        ),
    ),
    task_id: str | None = typer.Option(
        None,
        help="If passed, the agent will run only on this task.",
    ),
    with_evaluation: bool = typer.Option(
        True, help="If true, the evaluation will be run after the experiment run."
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "JSON string, if passed, the experiment config with be deeply "
            'overridden with values from this JSON. E.g., \'{"config": {"dataset": "dev"}}\'.'
        ),
    ),
    clear_first: bool = typer.Option(
        False, help="If true, the experiment output directory will be removed before the run."
    ),
    num_processes: int | None = typer.Option(
        1,
        help="The number of processes to use for parallel execution. Set to -1 to use all available CPUs.",
    ),
    process_index: int | None = typer.Option(
        None,
        help="The process index to be used for this run. It is set automatically when num-processes > 1. ",
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Development] Run an AppWorld agent from the appworld-agents library.
    """
    from appworld.common.imports import ensure_package_installed
    from appworld.common.io import jsonnet_load
    from appworld.task import load_task_ids

    update_root(root)
    ensure_package_installed("appworld_agents")
    verify_data_exists(with_setup=with_setup)
    verify_fully_installed(with_setup=with_setup)

    if experiment_name == "options":
        run_options_str, run_options_tree = get_run_options(
            path_store.experiment_configs, use_color=True, include_path_in_label=True
        )
        rprint(run_options_tree or run_options_str)
        return

    if experiment_name != "auto" and (model_name or agent_name or dataset_name):
        raise Exception(
            "If 'experiment_name' is not 'auto', then "
            "'model_name', 'agent_name' and 'dataset_name' should not be provided."
        )
        return
    if (
        (experiment_name == "auto")
        and ("options" not in (model_name, agent_name, dataset_name))
        and (not (model_name or agent_name or dataset_name))
    ):
        raise Exception(
            "If 'experiment_name' is 'auto', then "
            "'model_name', 'agent_name' and 'dataset_name' must be provided. "
            "You can set any or all of them to 'options' to see all individual options."
        )
    if dry_run and experiment_name != "auto":
        raise Exception("The 'dry_run' option is only applicable if 'experiment_name' is 'auto'.")

    if experiment_name == "auto":
        from appworld.task import load_task_ids, valid_dataset_names
        from appworld_agents.configs._generator.agents import AGENT_NAME_TO_INFO
        from appworld_agents.configs._generator.models import MODEL_NAME_TO_INFO
        from appworld_agents.configs._generator.run import make_experiment_config

        models_path = os.path.join(
            path_store.experiment_configs, "_generator", "models", "{creator}.py"
        )
        agents_path = os.path.join(
            path_store.experiment_configs, "_generator", "agents", "__init__.py"
        )
        dataset_path = os.path.join(path_store.data, "datasets", "{dataset_name}.txt")
        return_ = False
        console = Console(force_terminal=True, color_system="truecolor")
        if model_name == "options" or (model_name and model_name not in MODEL_NAME_TO_INFO):
            table = Table(
                title=Text(f"Available models: Find or add new in: {models_path}", style="bold"),
                expand=True,
            )
            table.add_column("creator", style="magenta")
            table.add_column("provider", style="green")
            table.add_column("model_name", style="cyan", no_wrap=True)
            for model_name_, model_info_ in MODEL_NAME_TO_INFO.items():
                table.add_row(model_info_.creator, model_info_.provider, model_name_)
            console.print(table)
            return_ = True
        if agent_name == "options" or (agent_name and agent_name not in AGENT_NAME_TO_INFO):
            table = Table(
                title=Text(f"Available agents: Find or add new in: {agents_path}", style="bold"),
                expand=True,
            )
            table.add_column("agent_name", style="cyan", no_wrap=True)
            table.add_column("agent_humanized_name", style="magenta")
            for agent_name_, agent_info_ in AGENT_NAME_TO_INFO.items():
                table.add_row(agent_name_, agent_info_.agent_humanized_name)
            console.print(table)
            return_ = True
        valid_dataset_names_ = valid_dataset_names()
        if dataset_name == "options" or (dataset_name and dataset_name not in valid_dataset_names_):
            table = Table(
                title=Text(f"Available datasets: Find or add new in: {dataset_path}", style="bold"),
                expand=True,
            )
            table.add_column("dataset_name", style="cyan", no_wrap=True)
            table.add_column("number_of_tasks", style="magenta")
            for dataset_name_ in valid_dataset_names_:
                task_ids = load_task_ids(dataset_name_)
                table.add_row(dataset_name_, str(len(task_ids)))
            console.print(table)
            return_ = True
        if return_:
            return
        if not agent_name:
            raise Exception(
                "The CLI argument 'agent-name' is required. You can set it to 'options' to see all options."
            )
        if not model_name:
            raise Exception(
                "The CLI argument 'model-name' is required. You can set it to 'options' to see all options."
            )
        if not dataset_name:
            raise Exception(
                "The CLI argument 'dataset-name' is required. You can set it to 'options' to see all options."
            )
        agent_info = AGENT_NAME_TO_INFO[agent_name]
        model_info = MODEL_NAME_TO_INFO[model_name]
        output = make_experiment_config(
            model_name=model_info.model_name,
            agent_name=agent_info.agent_name,
            dataset_name=dataset_name,
            save=True,
            dry_run=dry_run,
        )
        experiment_name = output["name"]
        experiment_file_path = path_store.experiment_config_file_path(experiment_name)
        print(f"Experiment config generated and saved at: {experiment_file_path}")
        if dry_run:
            return

    experiment_file_path = path_store.experiment_config_file_path(experiment_name)
    if not os.path.exists(experiment_file_path):
        raise Exception(f"The experiment config file {experiment_file_path} does not exist.")
    experiment_config = jsonnet_load(
        experiment_file_path,
        APPWORLD_EXPERIMENT_PROMPTS_PATH=path_store.experiment_prompts,
        APPWORLD_EXPERIMENT_CONFIGS_PATH=path_store.experiment_configs,
        APPWORLD_EXPERIMENT_CODE_PATH=path_store.experiment_code,
    )
    backup_experiment_config = deepcopy(experiment_config)
    override = override or "{}"
    try:
        override_config: dict[str, Any] = json.loads(override)
    except json.JSONDecodeError as exception:
        raise Exception("'override' argument is not a valid JSON!") from exception
    override_dict(experiment_config, override_config)
    runner_type = experiment_config.pop("type")
    runner_config = experiment_config.pop("config")
    experiment_config.pop("metadata", None)  # only for reference, not used in code.

    if experiment_config:
        raise Exception(f"Unexpected keys in the experiment config: {experiment_config}")

    model_server_config = runner_config.pop("model_server", {"enabled": False})
    if process_index is None:
        model_server_config["log_file_path"] = os.path.join(
            path_store.experiment_outputs, experiment_name, "model_server.log"
        )
        model_server = BackgroundServer(**model_server_config)
        model_server.start()
        if model_server_config:
            model_server_config["started"] = True
        os.environ["MODEL_SERVER_URL"] = str(model_server.url) if model_server.enabled else ""

    allowed_runner_types_ = allowed_runner_types()
    if runner_type not in allowed_runner_types_:
        raise Exception(
            f"Invalid runner type: '{runner_type}'. "
            f"Your APPWORLD_ROOT is set to '{os.path.abspath(path_store.root)}'. "
            f"And within '{os.path.abspath(path_store.experiment_code)}' found following "
            f"sub-modules with runners (run.py): {allowed_runner_types_}."
        )
    run_module = importlib.import_module(f"appworld_agents.code.{runner_type}.run")
    if process_index is None:
        dataset_name = cast(str, run_module.extract_dataset_name(runner_config))
        task_ids = load_task_ids(dataset_name)
        if num_processes == -1:
            num_processes = min(
                len(task_ids),
                max(os.cpu_count() or 2, 2) - 1,  # One reserved for the main process
            )
            num_processes = max(num_processes, 1)
        assert num_processes is not None  # mypy
        num_processes = min(num_processes, len(task_ids))
        if clear_first:
            experiment_output_directory = os.path.join(
                path_store.experiment_outputs, experiment_name
            )
            shutil.rmtree(experiment_output_directory, ignore_errors=True)
        experiment_output_configs_directory = os.path.join(
            path_store.experiment_outputs, experiment_name, "configs"
        )
        os.makedirs(experiment_output_configs_directory, exist_ok=True)
        experiment_output_config_json_file_path = os.path.join(  # backup rendered config.
            experiment_output_configs_directory, f"{dataset_name}.json"
        )
        write_json(backup_experiment_config, experiment_output_config_json_file_path, silent=True)
        experiment_output_config_jsonnet_file_path = os.path.join(  # backup raw config.
            experiment_output_configs_directory, f"{dataset_name}.jsonnet"
        )
        shutil.copyfile(experiment_file_path, experiment_output_config_jsonnet_file_path)

    assert isinstance(num_processes, int)  # mypy

    def cross_platform_kill(processes: list[subprocess.Popen[Any]]) -> None:
        if sys.platform != "win32":
            os.killpg(os.getpgrp(), signal.SIGTERM)
            return
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

    if num_processes > 1 and process_index is None:
        if sys.platform != "win32":
            os.setpgrp()
        try:
            processes = []
            messages = []
            assert isinstance(num_processes, int)  # mypy
            for process_index_ in range(num_processes):
                arguments = [sys.executable, *sys.argv, "--process-index", str(process_index_)]
                process = subprocess.Popen(
                    arguments, env=os.environ, stderr=subprocess.PIPE, text=True
                )
                processes.append(process)
                messages.append(
                    f"🚀 Starting process {process_index_ + 1}/{num_processes} with process ID: {process.pid}"
                )
            rprint(
                Panel(
                    "\n".join(messages),
                    title="[bold]Process Launch Summary",
                    width=TERMINAL_LOGGER_WIDTH,
                )
            )
            exit_codes: list[int] = []
            stderr_logs: dict[int, str] = {}
            for process in processes:
                log_lines = []
                if process.stderr is not None:
                    for line in process.stderr:
                        sys.stderr.write(line)
                        log_lines.append(line)
                    process.stderr.close()
                stderr_logs[process.pid] = "".join(log_lines)
                exit_codes.append(process.wait())
            failed_processes = []
            failed_exit_codes = []
            for process, exit_code in zip(processes, exit_codes, strict=True):
                if exit_code != 0:
                    failed_processes.append(process)
                    failed_exit_codes.append(exit_code)
            if failed_processes:
                for process, code in zip(failed_processes, failed_exit_codes, strict=True):
                    error_log = stderr_logs.get(process.pid, "").strip()
                    header = f"[bold red]Process PID {process.pid} exited with {code}[/bold red]"
                    body = error_log or "(no stderr captured)"
                    rprint(Panel(body, title=header, width=TERMINAL_LOGGER_WIDTH))
                message = (
                    f"\n❌ Out of {num_processes} processes, {len(failed_processes)} "
                    f"have failed with exit codes: {failed_exit_codes}. Exiting."
                )
                rprint(message)
                cross_platform_kill(processes)
                sys.exit(failed_exit_codes[0])
        except KeyboardInterrupt:
            rprint("\n🛑 KeyboardInterrupt received. Killing the entire process group.")
            cross_platform_kill(processes)
            raise
    else:
        run_module.run_experiment(
            experiment_name=experiment_name,
            runner_config=deepcopy(runner_config),
            task_id=task_id,
            num_processes=num_processes,
            process_index=process_index,
        )
    if process_index is None and with_evaluation:
        if task_id is not None:
            dataset_name = None
        evaluate(
            experiment_name=experiment_name, dataset_name=dataset_name, task_id=task_id, root=root
        )


@app.command()
def evaluate(
    experiment_name: str = typer.Argument(
        ...,
        help=(
            "The name of the experiment to evaluate. "
            "The experiment output should be in ./experiments/outputs/{experiment_name} "
            "relative to the root directory."
        ),
    ),
    dataset_name: str | None = typer.Argument(
        None,
        help=(
            "The name of the dataset to evaluate the experiment on. "
            "The dataset should be in ./data/datasets/{dataset_name}.txt "
            "relative to the root directory. If not passed, "
            "the dataset used in the experiment config will be used."
        ),
    ),
    task_id: str | None = typer.Option(
        None,
        help=(
            "If passed, the evaluation will run on only this task. "
            "If passed, dataset_name must not be passed."
        ),
    ),
    with_setup: bool = typer.Option(
        False,
        "--with-setup/--without-setup",
        help=(
            "With setup enabled, the 'appworld download data' and 'appworld install' will be "
            "automatically run, if not already done."
        ),
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Development] Run experiment evaluation.
    """
    from appworld.evaluator import evaluate_dataset

    update_root(root)
    verify_data_exists(with_setup=with_setup)
    verify_fully_installed(with_setup=with_setup)
    if dataset_name is None and task_id is None:
        dataset_name = extract_dataset_name(experiment_name)
    evaluate_dataset(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        task_id=task_id,
        suppress_errors=True,
        include_details=True,
        aggregate_only=False,
        save_reports=True,
        print_report=True,
    )


def print_text(text: str, style: str) -> None:
    rprint(Text(text, style=style))


def print_code(code: str, language: str) -> None:
    rprint(Syntax(code, language, theme="monokai", line_numbers=False))


def show_serve_multiple_help(port: int, remote_docker: bool) -> None:
    # TODO: Fill this in.
    pass


def show_serve_environment_help(port: int, remote_docker: bool) -> None:
    print_text("🚀 Starting AppWorld Environment Server!\n", style="bold green")

    print_rule()
    print_text(
        " 📌 This is an HTTP interface to the AppWorld Python environment class.\n"
        + " 📌 You should not need it in most cases, but may in two:",
        style="bold cyan",
    )
    print_rule()

    # Usage 1:
    print()
    print_rule("Use-case 1.")
    this_line = "these 2 lines" if remote_docker else "this line"
    print_text(
        "\n👉 1a. You want to delegate code execution to a different machine, process, or even a Docker container. "
        + "\n👉 1b. E.g., for deployment or containarization for better code execution safety."
        + f"\n👉 1c. If so, add {this_line} to AppWorld initialization, everything else remains the same!\n",
        style="bold cyan",
    )
    docker_line = "\n            remote_docker=True  # <--" if remote_docker else ""
    code = dedent(f"""
        with AppWorld(
            task_id=task_id,
            remote_environment_url="http://0.0.0.0:{port}",  # <--{docker_line}
            ...
        ) as world:
            ...
    """)
    code = indent(code, "  ")
    print_code(code, "python")
    print_text(
        "\n👉 1d. If you do, we recommend you run end-to-end task verification first in the remote environment mode.\n",
        style="bold cyan",
    )
    docker_string = " --remote-docker" if remote_docker else ""
    code = dedent(f"""
        appworld verify tasks --remote-environment-url "http://0.0.0.0:{port}"{docker_string}
    """)
    code = indent(code, "  ")
    print_code(code, "python")

    print()
    print_rule("Use-case 2.")
    print_text(
        "\n👉 2a. You do not want to use the AppWorld Python class directly but want use it via HTTP calls. "
        + "\n👉 2b. E.g., if your code is in a different language or your dependencies are conflicting with ours."
        + "\n👉 2c. If so, read its documentation in /redoc and interact by making HTTP requests accordingly.\n",
        style="bold cyan",
    )
    code = dedent(f"""
        open http://0.0.0.0:{port}/redoc  # or /docs for FastAPI's builtin documentation
        curl -X POST http://0.0.0.0:{port}/initalize -d '{{...}}'  # Or any other HTTP client.
    """)
    code = indent(code, "  ")
    print_code(code, "bash")
    print_text(
        "\n👉 2d. If you do, we recommend writing a task verification loop like ours as a sanity check.\n",
        style="bold cyan",
    )
    print()


def show_serve_apis_help(port: int, remote_docker: bool) -> None:
    print_text("🚀 Starting AppWorld API Server!", style="bold green")

    print_rule()
    print_text(
        " 📌 This is an HTTP interface to the APIs in the AppWorld apps.\n"
        + " 📌 You should not need it in most cases, but may in a few:",
        style="bold cyan",
    )
    print_rule()
    # Usage 1:
    print()
    print_rule("Use-case 1.")
    this_line = "these 3 lines" if remote_docker else "these 2 lines"
    print_text(
        "\n👉 1a. Your agent writes Python code but uses a different HTTP client instead of our apis or requester modules."
        + f"\n👉 1b. If so, add {this_line} to AppWorld initialization, and import/use required module in execute(...)."
        + "\n👉 1c. It will handle all state management, e.g., loading task DB, saving outputs, etc.\n",
        style="bold cyan",
    )
    docker_line = "\n            remote_docker=True  # <--" if remote_docker else ""
    code = dedent(f"""
        remote_apis_url = "http://0.0.0.0:{port}"  # <--
        with AppWorld(
            task_id=task_id,
            remote_apis_url=remote_apis_url,  # <--{docker_line}
            ...
        ) as world:
            # Example usage:
            world.execute('''
            import requests # or httpx, etc.
            remote_apis_url = '{{remote_apis_url}}'
            ''') # ^ needed only once for each task world.
            world.execute("requests.get(remote_apis_url + '/spotify/songs/1')")
            # world.execute("apis.spotify.show_song(song_id=1)") # Our client continues to work.
            ...
    """)
    code = indent(code, "  ")
    print_code(code, "python")
    print_text(
        "\n👉 1d. If you do, we recommend you run end-to-end task verification first in the remote APIs mode.\n",
        style="bold cyan",
    )
    docker_string = " --remote-docker" if remote_docker else ""
    code = dedent(f"""
        appworld verify tasks --remote-apis-url "http://0.0.0.0:{port}"{docker_string}
    """)
    code = indent(code, "  ")
    print_code(code, "python")
    # Usage 2:
    print()
    print_rule("Use-case 2.")
    print_text(
        "\n👉 2a. You want your agent to write code in a non-Python language (e.g., bash scripts with curl HTTP client)."
        + "\n👉 2b. If so, handle state management yourself by making a few HTTP calls to "
        "our state-management APIs.\n",
        style="bold cyan",
    )
    code = dedent(f"""
        open http://0.0.0.0:{port}/redoc  # or /docs for documentation of state-management APIs.
        curl -X POST http://0.0.0.0:{port}/dbs -d '{{...}}'  # loading task DBs; Use any HTTP client.
    """)
    code = indent(code, "  ")
    print_code(code, "bash")
    # Usage 3:
    print()
    print_rule("Use-case 3.")
    print_text(
        "\n👉 3a. You want to see the FastAPI documentation of the APIs in the AppWorld apps.",
        style="bold cyan",
    )
    print_text(
        "👉 3b. If so, explore it at '/{app_name}/redoc' or '/{app_name}/docs'. E.g.,\n",
        style="bold cyan",
    )
    code = dedent(f"""
        open http://0.0.0.0:{port}/spotify/redoc  # FastAPI's inbuilt API docs explorer.
        open https://appworld.dev/api-explorer  # Our standardized API docs explorer.
    """)
    code = indent(code, "  ")
    print_code(code, "bash")
    # Usage 4:
    print()
    print_rule("Use-case 4.")
    print_text(
        "\n👉 4a. You want to explore AppWorld by making direct HTTP calls.",
        style="bold cyan",
    )
    print_text(
        "👉 4b. If so, pick your favorite HTTP client and make calls as per the documentation. E.g., \n",
        style="bold cyan",
    )
    code = dedent(f"""
        curl -X GET http://0.0.0.0:{port}/spotify/songs/1
    """)
    code = indent(code, "  ")
    print_code(code, "bash")
    print_text(
        "\n👉 4c. The server loads Base DB, on start, but "
        + "changes to task DB on AppWorld(...) initialization.\n",
        style="bold cyan",
    )


def show_serve_mcp_help(port: int, remote_docker: bool) -> None:
    # TODO: Fill this in.
    # NOTE: skip it for stdio transport as it interferes with stdio.
    pass


def show_serve_uis_help(port: int, remote_docker: bool) -> None:
    # TODO: Fill this in.
    pass


def allowed_runner_types() -> list[str]:
    runner_types: list[str] = []
    for root, _, file_names in os.walk(path_store.experiment_code):
        for file_name in file_names:
            if file_name == "run.py":
                file_path = os.path.join(root, file_name)
                with open(file_path) as file:
                    if "def run_experiment" in file.read():
                        relative_path = os.path.relpath(root, path_store.experiment_code)
                        assert isinstance(relative_path, str)  # mymy
                        runner_types.append(relative_path.replace(os.sep, "."))
    return runner_types


@app.command()
def pack(
    experiment_name: str = typer.Argument(
        ..., help="Experiment name to pack for the leaderboard submission."
    ),
    dataset_name: DatasetName = typer.Argument(
        ..., help="The dataset name associated with the experiment."
    ),
    method_name: str = typer.Argument(
        ..., help="The method name used in the experiment, e.g., ReAct."
    ),
    method_tooltip: str = typer.Argument(
        ..., help="A brief additional note about the method, e.g., Reason + Act."
    ),
    llm_name: str = typer.Argument(..., help="The LLM name used in the experiment, e.g., GPT4-o."),
    llm_tooltip: str = typer.Argument(
        ..., help="A brief additional note about the LLM, e.g., gpt-4o-2024-05-13."
    ),
    url: str = typer.Argument(
        ..., help="URL to find more information about this submission. E.g., GitHub repo."
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Leaderboard] Pack your experiment output for the leaderboard submission.
    """
    from appworld.leaderboard import pack_experiment

    update_root(root)
    pack_experiment(
        experiment_name=experiment_name,
        dataset_name=dataset_name.value,
        method_name=method_name,
        method_tooltip=method_tooltip,
        llm_name=llm_name,
        llm_tooltip=llm_tooltip,
        url=url,
    )


@app.command()
def unpack(
    experiment_name: str = typer.Argument(
        ..., help="Experiment name packed for the leaderboard submission."
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Leaderboard] Unpack an experiment output from the leaderboard submission.
    """
    from appworld.leaderboard import unpack_experiment

    update_root(root)
    unpack_experiment(experiment_name=experiment_name)


@app.command()
def make(
    testn_experiment_name: str = typer.Argument(
        ..., help="Experiment name for agent runs on the test_normal dataset."
    ),
    testc_experiment_name: str = typer.Argument(
        ..., help="Experiment name for agent runs on the test_challenge dataset."
    ),
    save: bool = typer.Option(
        False,
        help="Whether to save the leaderboard entry to the leaderboard file or just show it",
    ),
    root: str | None = typer.Option(
        ".",
        help=(
            "The AppWorld root directory. Pass only if the "
            "root directory is not the current directory."
        ),
    ),
) -> None:
    """
    [Leaderboard] Make a leaderboard entry from your agent's experiment outputs.
    """
    from appworld.leaderboard import make_leaderboard_entry

    update_root(root)
    leaderboard_entry = make_leaderboard_entry(
        testn_experiment_name=testn_experiment_name,
        testc_experiment_name=testc_experiment_name,
        save=save,
    )
    rprint(leaderboard_entry)


def version_callback(value: bool) -> None:
    if value:
        print(appworld.__version__)
        raise typer.Exit()


@app.callback()
def version(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Print the appworld package version and exit.",
    ),
) -> None:
    pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
