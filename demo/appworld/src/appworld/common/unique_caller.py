from collections.abc import Callable
from typing import Any


class UniqueCaller:
    """
    A function caller that keeps track of the returned value, and keeps calling the function
    until it returns a value that hasn't been returned before. This is useful in combination
    with Faker (when fake.unique.xyz is not sufficient). The generated values need to be hashable.
    """

    def __init__(self) -> None:
        self.values: set[Any] = set()

    def __call__(
        self, function: Callable[..., Any], *args: Any, max_calls: int = 10, **kwargs: Any
    ) -> Any:
        for _ in range(max_calls):
            value = function(*args, **kwargs)
            if value not in self.values:
                self.values.add(value)
                return value
        raise Exception(f"Could not find a unique value after {max_calls} calls.")
