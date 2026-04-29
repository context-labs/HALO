from typing import TypeVar


T = TypeVar("T", bound=type)


def deep_copy_class(klass: T, name: str | None = None) -> T:
    # https://stackoverflow.com/questions/9541025/how-to-copy-a-class
    name = name or klass.__name__
    metacls = type(klass)
    return metacls(name, klass.__bases__, dict(klass.__dict__))
