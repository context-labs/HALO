from datetime import datetime
from typing import Any, Literal, Self, TypeVar, cast
from typing import get_args as get_type_args  # noqa

from munch import Munch
from typing_extensions import TypedDict

from appworld.common.registrable import Registrable as _Registrable


# NOTE: If I add list[float | int | str] to AnswerType,
# it has to be added at the end. It's the
# order in which pydantic parses the types.
AnswerType = None | float | int | str
Dict = dict[str, Any]
ListOfDicts = list[dict[str, Any]]
ListOfStrs = list[str]
ListOfNums = list[int | float]
ListOrDict = list[dict[str, Any]] | dict[str, Any] | list[str | int | float]
TItem = TypeVar("TItem")
TValue = TypeVar("TValue")
TNumber = TypeVar("TNumber", int, float)
Number = int | float
Comparable = int | float | datetime


class NotificationDict(TypedDict):
    type: str
    title: str
    message: str
    data: dict[str, int]


class TypedMunch(Munch):  # type: ignore
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)
        self._apply_type_hints()

    def _apply_type_hints(self) -> None:
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = TypedMunch(value)
            elif isinstance(value, list):
                self._apply_type_hints_to_list(value)

    def _apply_type_hints_to_list(self, lst) -> None:  # type: ignore
        for i, value in enumerate(lst):
            if isinstance(value, dict):
                lst[i] = TypedMunch(value)


class CustomErrorMessageMunch(Munch):  # type: ignore
    error_message: str  # has to be class var, o/w it shows up as dict key.

    def __init__(self, error_message: str, *args, **kwargs) -> None:  # type: ignore
        # error_message should have a "{key}" in it which will be replaced by the
        # the key that was not found.
        super().__init__(*args, **kwargs)
        self.__class__.error_message = error_message

    @classmethod
    def build(cls, error_message: str, *args: Any, **kwargs: Any) -> Self:
        class _CustomErrorMessageMunch(cls):  # type: ignore
            pass

        output = _CustomErrorMessageMunch(error_message, *args, **kwargs)
        return cast(Self, output)

    def __getattr__(self, key: str) -> Any:
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, key)
        except AttributeError:
            from appworld.apps.lib.apis.authentication import raise_http_exception

            try:
                return self[key]
            except KeyError:
                error_message = self.__class__.error_message.format(key=key)
                raise_http_exception(error_message)


class TestData(TypedDict):
    requirement: str
    label: Literal["no_op_fail", "no_op_pass"]


class Registrable(_Registrable):
    # https://github.com/tiangolo/sqlmodel/pull/256
    # To get SQLModel to be registrable. Temporary workaround.
    __config__ = None

    @classmethod
    def registered_name(cls, subclass: type[Self]) -> str | None:
        for name, klass in cls._registry[cls].items():
            if subclass == klass:
                return name
        return None


class FromDict(Registrable):
    """
    A super simplified version of Allennlp's FromParams, which doesn't
    recursively build nested objects, just the root-level one.
    """

    @classmethod
    def from_dict(cls, dict_: dict[str, Any]) -> Self:
        if "type" not in dict_:
            raise Exception("Missing 'type' in config dict.")
        class_type = dict_.pop("type")
        if not issubclass(cls, Registrable):
            raise Exception(
                f"The class {cls} is not a subclass of Registrable. Can't use from_dict without it."
            )
        if not cls.is_registered(class_type):
            raise Exception(
                f"The base class {cls} does not have any registered class as {class_type}."
            )
        sub_class = cls.by_name(class_type)
        instance = sub_class(**dict_)
        dict_["type"] = class_type  # add it back as it's mutable.
        instance._registered_name: str = class_type  # type: ignore
        return instance


def cast_dict(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise Exception(f"Expected a dict. Found {type(data)}.")
    return cast(dict[str, Any], data)
