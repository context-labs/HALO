import os
from typing import cast

from jinja2 import BaseLoader, Environment

from appworld.common.io import read_file


def apply_chat_template(messages: list[dict[str, str]], template_file_path_or_content: str) -> str:
    if os.path.exists(template_file_path_or_content):
        template_file_path_or_content_ = read_file(template_file_path_or_content)
        template_file_path_or_content = cast(str, template_file_path_or_content_)
    jinja_template = Environment(loader=BaseLoader()).from_string(template_file_path_or_content)
    output = jinja_template.render(messages=messages)
    assert isinstance(output, str)  # mypy
    return output
