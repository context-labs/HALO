import inspect
import sys
import textwrap
from collections.abc import Callable
from typing import Any, Literal, cast

from rich import print as _rprint
from rich.console import Console
from rich.rule import Rule
from rich.syntax import Syntax
from rich.traceback import install as rich_traceback_install


rprint: Callable[..., None] = cast(Callable[..., None], _rprint)


def indented_print(string_to_print: str) -> None:
    """
    Looks at the indentation present at the line this function
    is called, and indents the string_to_print by that much.
    This is useful in data generation scripts where there is a lot of nesting
    and I need to keep track of the progress.
    """
    currentframe = inspect.currentframe()
    if currentframe is None:
        print(string_to_print)
        return

    frame = currentframe.f_back
    if frame is None:
        print(string_to_print)
        return

    frame_info = inspect.getframeinfo(frame)
    if frame_info is None:
        print(string_to_print)
        return

    caller_line = frame_info.lineno
    caller_file = frame_info.filename

    with open(caller_file, encoding="utf-8") as file:
        for line_num, line_content in enumerate(file, start=1):
            if line_num == caller_line:
                num_prefix_spaces = len(line_content) - len(line_content.lstrip(" "))
                prefix_spaces = " " * num_prefix_spaces
                print(textwrap.indent(string_to_print, prefix_spaces))
                break


def print_rule(title: str = "", console: Console | None = None) -> None:
    function = rprint if console is None else console.print
    if sys.platform == "win32":  # Temporary fix for windows
        function(f"------- {title} -------")
    else:
        function(Rule(title=title))


def print_code(
    code: str,
    language: str = "python",
    theme: Literal["monokai", "none"] = "monokai",
    line_numbers: bool = True,
    console: Console | None = None,
) -> None:
    if theme == "none":
        if console is None:
            print(code)
        else:
            console.print(code)
        return
    if console is None:
        rprint(Syntax(code, language, theme=theme, line_numbers=line_numbers))
    else:
        console.print(Syntax(code, language, theme=theme, line_numbers=line_numbers))


def color_traceback() -> None:
    rich_traceback_install(width=120)
    from appworld.requester import add_pre_excepthook  # to avoid circular import.

    add_pre_excepthook()


def table_data_to_string(table_data: dict[str, list[Any]]) -> str:
    """
    Example table_data:
    data = {
        "header": ["row1", "row2"],
        "col1": [10.0, 20.0],
        "col2": [30.0, 40.0]
    }
    output:
      header |     col1     |    col2
    ---------+--------------+-----------
        row1 |    10.0      |    30.0
        row2 |    20.0      |    40.9
    """
    headers = list(table_data.keys())
    rows = list(zip(*table_data.values(), strict=True))
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows, strict=True)]

    def center_align(text: str, width: int) -> str:
        return text.center(width)

    table_string = ""
    header_row = " | ".join(
        f"{center_align(str(item), col_widths[i])}" for i, item in enumerate(headers)
    )
    table_string = header_row + "\n"
    table_string += ("-+-".join("-" * col_widths[i] for i in range(len(headers)))) + "\n"
    for row in rows:
        table_string += (
            " | ".join(f"{center_align(str(item), col_widths[i])}" for i, item in enumerate(row))
        ) + "\n"
    return table_string
