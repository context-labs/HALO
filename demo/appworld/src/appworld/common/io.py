import copy
import csv
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterator
from datetime import date, datetime
from typing import Any, Literal, cast
from typing import get_args as get_type_args  # noqa: F401 (better name)

import orjson
import yaml
from munch import unmunchify
from pendulum import Date as _Date

from appworld.common.naming import cached_pluralize as cached_pluralize
from appworld.common.naming import cached_singularize as cached_singularize
from appworld.common.naming import cached_underscore as cached_underscore
from appworld.common.naming import inflect as inflect
from appworld.common.naming import is_plural as is_plural
from appworld.common.naming import modelize as modelize


FILE_LOGGER_WIDTH = 200
TERMINAL_LOGGER_WIDTH: int | None = None if sys.stdout.isatty() else FILE_LOGGER_WIDTH
if TERMINAL_LOGGER_WIDTH is not None:
    os.environ["TERMINAL_WIDTH"] = str(  # This is what rich (via typer) uses for width.
        TERMINAL_LOGGER_WIDTH
    )


def file_locked(function: Callable[..., Any], timeout: int = 10, **kwargs: Any) -> Any:
    from appworld.common.imports import ensure_package_installed

    ensure_package_installed("filelock")
    from filelock import FileLock, Timeout

    if "file_path" not in kwargs:
        raise ValueError("file_path must be provided in a file_locked function.")
    lock_file_path = kwargs["file_path"] + ".lock"
    lock = FileLock(lock_file_path, timeout=timeout)
    try:
        with lock:
            return function(**kwargs)
    except Timeout:
        print(f"Couldn't acquire the lock for {lock_file_path}.")


def write_csv(instances: list[dict[str, Any]], file_path: str) -> None:
    if not instances:
        raise Exception("No csv instances to write.")
    csv_field_names = list(instances[0].keys())
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_field_names)
        writer.writeheader()
        for instance in instances:
            writer.writerow(instance)


def read_jsonl(file_path: str, use_json_plus: bool = True) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    json_: Any = orjson
    mode = "rb"
    if use_json_plus:
        json_ = JsonPlus
        mode = "r"
        kwargs["encoding"] = "utf-8"
    with open(file_path, mode, **kwargs) as file:
        instances = [json_.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def yield_jsonl(file_path: str, use_json_plus: bool = True) -> Iterator[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    json_: Any = orjson
    mode = "rb"
    if use_json_plus:
        json_ = JsonPlus
        mode = "r"
        kwargs["encoding"] = "utf-8"
    with open(file_path, mode, **kwargs) as file:
        for line in file:
            yield json_.loads(line.strip())


def write_jsonl(
    instances: list[dict[str, Any]],
    file_path: str,
    silent: bool = False,
    use_json_plus: bool = True,
    append: bool = False,
) -> None:
    if not silent:
        print(f"Writing {len(instances)} json instances in {file_path}")
    json_: Any = orjson
    decode: Callable[[Any], str] = lambda x: cast(bytes, x).decode("utf-8")
    if use_json_plus:
        json_ = JsonPlus
        decode = lambda x: cast(str, x)
    mode = "a" if append else "w"
    with open(file_path, mode, encoding="utf-8") as file:
        for instance in instances:
            file.write(decode(json_.dumps(instance)) + "\n")


def read_json(file_path: str, use_json_plus: bool = True) -> dict[str, Any]:
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    if not content.strip():
        return {}
    json_: Any = JsonPlus if use_json_plus else orjson
    data = json_.loads(content)
    return cast(dict[str, Any], data)


def write_json(
    data: dict[str, Any],
    file_path: str,
    silent: bool = False,
    use_json_plus: bool = True,
    indent: int = 4,
) -> None:
    # orjson does not support indent level = 4, so using plain json.
    if not silent:
        print(f"Writing json in {file_path}")
    json_: Any = JsonPlus if use_json_plus else json
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(json_.dumps(data, indent=indent))


def update_json(
    data: dict[str, Any], file_path: str, silent: bool = False, use_json_plus: bool = True
) -> None:
    # orjson does not support indent level = 4, so using plain json.
    if not silent:
        print(f"Updating json in {file_path}")
    existing_data = {}
    if os.path.exists(file_path):
        existing_data = read_json(file_path, use_json_plus=use_json_plus)
    for key, value in data.items():
        existing_data[key] = value
    write_json(existing_data, file_path, use_json_plus=use_json_plus, silent=silent)


def read_file(file_path: str, mode: Literal["r", "rb"] = "r") -> str | bytes:
    if "b" in mode:
        with open(file_path, mode=mode) as file:
            content_bytes = file.read()
        return cast(bytes, content_bytes)
    else:
        with open(file_path, mode=mode, encoding="utf-8") as file:
            content_str = file.read()
        return cast(str, content_str)


def write_file(content: str | bytes, file_path: str, mode: str = "w") -> None:
    if "b" in mode:
        with open(file_path, mode) as file:
            file.write(cast(bytes, content))
    else:
        with open(file_path, mode, encoding="utf-8") as file:
            file.write(cast(str, content))


def append_file(content: str, file_path: str, skip_empty: bool = True) -> None:
    if skip_empty and not content.strip():
        return
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(content)


def touch_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        write_file("", file_path)


def read_toml(file_path: str) -> dict[str, Any]:
    import tomllib

    with open(file_path, mode="rb") as file:
        content = tomllib.load(file)
    return content


def safely_remove_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except (FileNotFoundError, PermissionError):  # needed in case of multiple processes
        pass


def safely_remove_directory(directory_path: str, create: bool = False) -> None:
    shutil.rmtree(directory_path, ignore_errors=True)
    if create:
        os.makedirs(directory_path, exist_ok=True)


def is_lfs_pointer(file_path: str) -> bool:
    """Check if the file is a Git LFS pointer."""
    try:
        with open(file_path, encoding="utf-8") as file:
            lines = [file.readline() for _ in range(3)]
        return (
            len(lines) >= 1
            and lines[0].startswith("version https://git-lfs.github.com/spec/")
            and any("oid sha256:" in line for line in lines)
            and any("size " in line for line in lines)
        )
    except UnicodeDecodeError:
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


def maybe_create_parent_directory(file_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)


def ensure_file(file_path: str, not_lfs_pointer: bool = True) -> None:
    if not os.path.exists(file_path):
        raise Exception(f"File {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise Exception(f"File {file_path} is not a file.")
    if not_lfs_pointer and is_lfs_pointer(file_path):
        raise Exception(f"File {file_path} is a Git LFS pointer and not a full file.")


def ensure_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        raise Exception(f"Directory {directory_path} does not exist.")
    if not os.path.isdir(directory_path):
        raise Exception(f"Path {directory_path} is not a directory.")


def safe_shutil_copy(source_file_path: str, destination_file_path: str, timeout: int = 5) -> None:
    from appworld.common.imports import ensure_package_installed

    ensure_package_installed("filelock")
    from filelock import FileLock, Timeout

    lock_path = f"{destination_file_path}.copy.lock"
    marker_path = f"{destination_file_path}.copy.done"
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=timeout):
            if os.path.exists(marker_path):
                return
            shutil.copy(source_file_path, destination_file_path)
            write_file("done", marker_path)
    except Timeout:
        pass


def copy_directory(source_directory: str, target_directory: str, delete_first: bool = True) -> None:
    if delete_first:
        shutil.rmtree(target_directory, ignore_errors=True)
    shutil.copytree(source_directory, target_directory)


def unix_basename(path: str, with_extension: bool = True) -> str:
    # cannot use os.basename for file_system app paths
    # as they are unix based and os.basename will break on windows.
    basename = path.rstrip("/").split("/")[-1]
    if not with_extension:
        basename = basename.split(".")[0]
    return basename


class JsonPlus:
    int_marker = "{_INT_}:"
    float_marker = "{_FLOAT_}:"
    datetime_marker = "{_DATETIME_}:"
    date_marker = "{_DATE_}:"

    @classmethod
    def serialize(cls, object_: dict[Any, Any] | list[Any]) -> None:
        if isinstance(object_, list):
            for index, item in enumerate(list(object_)):  # list is to copy.
                if isinstance(item, datetime):
                    object_[index] = cls.datetime_marker + item.isoformat()
                if isinstance(item, date) and not isinstance(item, datetime):
                    object_[index] = cls.date_marker + item.isoformat()
                # no need to deal with int, float as json already does that.
                if isinstance(item, dict):
                    cls.serialize(item)
                if isinstance(item, list):
                    cls.serialize(item)
        if isinstance(object_, dict):
            for key, value in list(object_.items()):
                if isinstance(value, datetime):  # this has to be first.
                    object_[key] = cls.datetime_marker + value.isoformat()
                if isinstance(value, date) and not isinstance(value, datetime):
                    object_[key] = cls.date_marker + value.isoformat()
                if isinstance(key, datetime):
                    object_[cls.datetime_marker + key.isoformat()] = object_.pop(key)
                if isinstance(key, date) and not isinstance(key, datetime):
                    object_[cls.date_marker + key.isoformat()] = object_.pop(key)
                if isinstance(key, int):
                    object_[cls.int_marker + str(key)] = object_.pop(key)
                if isinstance(key, float):
                    object_[cls.float_marker + str(key)] = object_.pop(key)
                if isinstance(value, dict):
                    cls.serialize(value)
                if isinstance(value, list):
                    cls.serialize(value)

    @classmethod
    def deserialize(cls, object_: dict[Any, Any] | list[Any]) -> None:
        from pendulum import Date

        from appworld.common.datetime import DateTime  # to avoid circular import

        if isinstance(object_, list):
            for index, item in enumerate(list(object_)):  # list is to copy.
                if isinstance(item, str) and item.startswith(cls.datetime_marker):
                    object_[index] = DateTime.fromisoformat(item.removeprefix(cls.datetime_marker))
                if isinstance(item, str) and item.startswith(cls.date_marker):
                    object_[index] = Date.fromisoformat(item.removeprefix(cls.date_marker))
                # no need to deal with int, float as json already does that.
                if isinstance(item, dict):
                    cls.deserialize(item)
                if isinstance(item, list):
                    cls.deserialize(item)

        if isinstance(object_, dict):
            for key, value in list(object_.items()):
                if isinstance(value, str) and value.startswith(
                    cls.datetime_marker
                ):  # this has to be first.
                    object_[key] = DateTime.fromisoformat(value.removeprefix(cls.datetime_marker))
                if isinstance(value, str) and value.startswith(cls.date_marker):
                    object_[key] = Date.fromisoformat(value.removeprefix(cls.date_marker))
                if key.startswith(cls.datetime_marker):
                    object_[DateTime.fromisoformat(key.removeprefix(cls.datetime_marker))] = (
                        object_.pop(key)
                    )
                if key.startswith(cls.date_marker):
                    object_[_Date.fromisoformat(key.removeprefix(cls.date_marker))] = object_.pop(
                        key
                    )
                if key.startswith(cls.int_marker):
                    object_[int(key.removeprefix(cls.int_marker))] = object_.pop(key)
                if key.startswith(cls.float_marker):
                    object_[float(key.removeprefix(cls.float_marker))] = object_.pop(key)
                if isinstance(value, dict):
                    cls.deserialize(value)
                if isinstance(value, list):
                    cls.deserialize(value)

    @classmethod
    def copy(
        cls, object_: dict[Any, Any] | list[Any] | tuple[Any, ...]
    ) -> dict[Any, Any] | list[Any]:
        if isinstance(object_, list | tuple):
            return [cls.copy(item) for item in object_]
        elif isinstance(object_, dict):
            return {key: cls.copy(value) for key, value in object_.items()}
        return copy.deepcopy(object_)

    @classmethod
    def dumps(cls, object_: dict[Any, Any] | list[Any] | tuple[Any, ...], **kwargs: Any) -> str:
        object_ = cls.copy(object_)
        cls.serialize(object_)
        return json.dumps(object_, **kwargs)

    @classmethod
    def loads(cls, json_str: str, **kwargs: Any) -> dict[Any, Any] | list[Any] | tuple[Any, ...]:
        object_: Any = json.loads(json_str, **kwargs)
        cls.deserialize(object_)
        return cast(dict[Any, Any] | list[Any] | tuple[Any, ...], object_)

    @classmethod
    def dump(
        cls, object_: dict[Any, Any] | list[Any] | tuple[Any, ...], fp: Any, **kwargs: Any
    ) -> None:
        object_ = cls.copy(object_)
        cls.serialize(object_)
        json.dump(object_, fp, **kwargs)

    @classmethod
    def load(cls, fp: Any, **kwargs: Any) -> dict[Any, Any] | list[Any]:
        object_: Any = json.load(fp, **kwargs)
        cls.deserialize(object_)
        return cast(dict[Any, Any] | list[Any], object_)


def dump_yaml(json_object: list[Any] | dict[Any, Any], indent: int = 2) -> str:
    json_object = unmunchify(json_object)
    return yaml.dump(json_object, sort_keys=False, width=float("inf"), indent=indent)


def load_yaml(yaml_str: str) -> list[Any] | dict[Any, Any]:
    try:
        return cast(list[Any] | dict[Any, Any], yaml.safe_load(yaml_str))
    except yaml.YAMLError as exception:
        message = "Could not parse string as yaml:" + "\n" + yaml_str
        raise yaml.YAMLError(f"Error loading yaml. {message}") from exception


def extract_yaml_section(yaml_string: str, section: str, keep_header: bool = False) -> str:
    lines = yaml_string.split("\n")
    section_lines: list[str] = []
    in_section = False
    for line in lines:
        if line.startswith(f"{section}:"):
            in_section = True
            continue
        if line.endswith(":") or line.endswith(": >"):
            in_section = False
            continue
        if in_section:
            section_lines.append(line)
    result = "\n".join(section_lines)
    if keep_header:
        result = f"{section}:\n{result}"
    return result


def jsonnet_loads(jsonnet_string: str, **kwargs: Any) -> dict[str, Any]:
    try:
        import _jsonnet
    except ImportError:  # installing it in windows is tricky.
        print("WARNING: jsonnet is not installed. Using json instead.")
        return cast(dict[str, Any], json.loads(jsonnet_string))
    output = json.loads(
        _jsonnet.evaluate_snippet("", jsonnet_string, ext_vars=dict(os.environ) | kwargs)
    )
    return cast(dict[str, Any], output)


def jsonnet_load(file_path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        import _jsonnet
    except ImportError:  # installing it in windows is tricky.
        print("WARNING: jsonnet is not installed. Using json instead.")
        return read_json(file_path)
    output = json.loads(_jsonnet.evaluate_file(file_path, ext_vars=dict(os.environ) | kwargs))
    return cast(dict[str, Any], output)
