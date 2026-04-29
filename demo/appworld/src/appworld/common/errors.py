import linecache
import os
import re
import traceback
from textwrap import dedent
from traceback import TracebackException
from types import TracebackType
from typing import Self


def get_stack_trace_from_exception(
    exception: BaseException,
    remove_cwd: bool = True,
    only_ipython: bool = False,
    add_http_exception_message: bool = True,
) -> str:
    from fastapi.exceptions import HTTPException  # saving expensive import so kept local.

    error_stack_trace = "".join(TracebackException.from_exception(exception).format())
    if remove_cwd:
        error_stack_trace = error_stack_trace.replace(os.getcwd(), ".")
    if only_ipython:
        lines = error_stack_trace.splitlines()
        filtered_lines = []
        take_lines = False
        for line in lines:
            if not line.startswith("  "):
                filtered_lines.append(line)
                continue
            match = re.compile(r' +File "(.*?)", line (\d+), in .+').match(line)
            if match:
                active_file_path = match.group(1)
                if "python-input" in active_file_path:
                    take_lines = True
                else:
                    take_lines = False
            if take_lines:
                filtered_lines.append(line)
        error_stack_trace = "\n".join(filtered_lines)
    if add_http_exception_message and isinstance(exception, HTTPException):
        if exception.detail not in error_stack_trace:
            error_stack_trace = error_stack_trace + ": " + exception.detail
    error_stack_trace = error_stack_trace.replace("fastapi.exceptions.HTTPException", "Exception")
    error_stack_trace = error_stack_trace.replace("Exception: 422:", "Exception:")
    # https://github.com/ipython/ipython/issues/12831
    ipython_issue_message = dedent("""
    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
    AttributeError: 'tuple' object has no attribute 'tb_frame'
    """).strip()
    error_stack_trace = error_stack_trace.replace(ipython_issue_message, "")
    return error_stack_trace


def get_traceback_stack_data(
    num_lines_before: int = 5, num_lines_after: int = 10
) -> list[dict[str, object]]:
    traceback_info: list[dict[str, object]] = []
    stack = traceback.extract_stack()
    for frame in stack:
        file_path = frame[0]  # Get the file path
        if not os.path.exists(file_path):
            continue
        file_line_number = frame[1]  # Get the line number
        function_name = frame[2]  # Get the function name
        file_line_index = file_line_number - 1  # Adjust the line index for 0-based indexing
        context_lines: list[str] = []
        context_line_index: int | None = None
        target_line: str | None = None
        for index in range(file_line_index - num_lines_before, file_line_index + num_lines_after):
            code_line = linecache.getline(file_path, index)
            if index == file_line_index + 1:
                target_line = code_line.strip()
                context_line_index = len(context_lines)
            if not code_line.strip():
                continue
            context_lines.append(code_line.rstrip())
        assert context_line_index is not None
        if target_line != context_lines[context_line_index].strip():
            print(
                "WARNING: Proper traceback for evaluation test could not be created. "
                "It only affect cosmetics of the task's evaluation report. "
                "Please raise an issue if you see this message."
            )
        traceback_info.append(
            {
                "file_path": file_path,
                "file_line_index": file_line_index,
                "context_line_index": context_line_index,
                "function_name": function_name,
                "context": "\n".join(context_lines),
            }
        )
    return traceback_info


class NoSuppressError(Exception):
    # A way to bypass the MaybeSuppressErrors context manager.
    pass


class MaybeSuppressErrors:
    """
    Usage:
    with MaybeSuppressErrors(suppress_errors=True) as suppressor:
        ...
        # some potentially error-raising code
    suppressor.raised_error  # True if an error was raised, False otherwise
    if something raises NoSuppressError, it will not be suppressed.
    """

    def __init__(self, suppress_errors: bool = True) -> None:
        self.suppress_errors = suppress_errors
        self.raised_error = False
        self.exception: BaseException | None = None

    def print_traceback(self, remove_cwd: bool = False) -> None:
        if self.raised_error:
            assert self.exception is not None
            print(get_stack_trace_from_exception(self.exception, remove_cwd=remove_cwd))

    def raise_error(self) -> None:
        if self.raised_error:
            assert self.exception is not None
            raise self.exception

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if isinstance(exc_value, NoSuppressError):
            raise exc_value
        if exc_type is not None and self.suppress_errors:
            self.raised_error = True
            self.exception = exc_value
            return True  # Suppress the error
        return False
