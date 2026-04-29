from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict, model_validator, validate_call
from typing_extensions import TypedDict

from appworld.common.collections import sorted_plus


if TYPE_CHECKING:
    from appworld.apps.lib.models.orm import TSQLModel
else:
    TSQLModel = Any


class StrictPydanticModel(PydanticModel):
    model_config = ConfigDict(extra="forbid")


class FixedFilterBy(StrictPydanticModel):
    value: Any
    ignore_value: Any
    is_date: bool
    is_uncased: bool
    is_singularized: bool


class FixedFilterByDict(TypedDict):
    value: Any
    ignore_value: Any
    is_date: bool
    is_uncased: bool
    is_singularized: bool


class MembershipFilterBy(StrictPydanticModel):
    value: Any
    ignore_value: Any
    is_date: bool


class MembershipFilterByDict(TypedDict):
    value: Any
    ignore_value: Any
    is_date: bool


class HasFilterBy(StrictPydanticModel):
    value: bool | None
    ignore_value: bool | None


class RangeFilterBy(StrictPydanticModel):
    min: Any
    max: Any
    ignore_min: Any
    ignore_max: Any
    is_date: bool


class HasFilterByDict(TypedDict):
    value: bool | None
    ignore_value: bool | None


class RangeFilterByDict(TypedDict):
    min: Any
    max: Any
    ignore_min: Any
    ignore_max: Any
    is_date: bool


class PageBy(StrictPydanticModel):
    index: int | None = None
    limit: int | None = None

    def defined(self) -> bool:
        return self.index is not None and self.limit is not None

    @model_validator(mode="before")
    def check_index_and_limit(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        index, limit = values.get("index"), values.get("limit")
        if (index is None and limit is not None) or (index is not None and limit is None):
            raise ValueError(
                "Either both or none of page_index and page_limit must be passed to retrieve function."
            )
        return values


class PageByDict(TypedDict, total=False):
    index: int | None
    limit: int | None


class SortBy(StrictPydanticModel):
    value: str | None = None
    valid_fields: Sequence[str] | None = None
    value_map: dict[Any, Any] | None = None
    last_resort: str | None = None  # when no ranking preference is given (no query, sort_by)

    def defined(self) -> bool:
        return (self.value is not None) or (self.last_resort is not None)


class SortByDict(TypedDict, total=False):
    value: str | None
    valid_fields: Sequence[str] | None
    value_map: dict[Any, Any] | None
    last_resort: str | None


class DictBy(StrictPydanticModel):
    humanize: bool = True
    name: str = "default"


class FilterBy(StrictPydanticModel):
    fixed: dict[str, FixedFilterBy] | None = None
    membership: dict[str, MembershipFilterBy] | None = None
    has: dict[str, HasFilterBy] | None = None
    range: dict[str, RangeFilterBy] | None = None

    def __init__(
        self,
        fixed: dict[str, Any] | None = None,
        membership: dict[str, Any] | None = None,
        has: dict[str, Any] | None = None,
        range: dict[str, Any] | None = None,
    ):
        super().__init__(  # type: ignore[call-arg]
            fixed={key: FixedFilterBy(**value) for key, value in (fixed or {}).items()},
            membership={
                key: MembershipFilterBy(**value) for key, value in (membership or {}).items()
            },
            has={key: HasFilterBy(**value) for key, value in (has or {}).items()},
            range={key: RangeFilterBy(**value) for key, value in (range or {}).items()},
        )

    def defined(self) -> bool:
        return any([self.fixed, self.membership, self.has, self.range])


class FilterByDict(TypedDict, total=False):
    fixed: dict[str, FixedFilterByDict]
    membership: dict[str, MembershipFilterByDict]
    has: dict[str, HasFilterByDict]
    range: dict[str, RangeFilterByDict]


class RoundtripBy(StrictPydanticModel):
    path: str | None = None
    operate_on_mapped: list[Literal["search", "filter", "sort", "return"]] | None = None

    def defined(
        self, operate: Literal["any", "search", "filter", "sort", "return"] = "any"
    ) -> bool:
        if operate == "any":
            return self.path is not None
        return operate in (self.operate_on_mapped or [])

    @model_validator(mode="before")
    def check_operate_on_mapped(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        path, operate_on_mapped = values.get("path"), values.get("operate_on_mapped")
        if (path is None) != (operate_on_mapped is None):
            raise ValueError(
                "Either both or none of path and operate_on_mapped must be passed to retrieve function."
            )
        return values


class RoundtripByDict(TypedDict, total=False):
    path: str
    operate_on_mapped: list[Literal["search", "filter", "sort", "return"]]


def _maybe_transform_field_data(
    field_is_date: bool, field_name: str, field_value: Any, ignore_value: Any
) -> tuple[Any, Any]:
    from appworld.apps.lib.apis.datetime import parse_date, parse_date_or_raise_error

    if field_is_date:
        field_value = parse_date_or_raise_error(field_value, field_name)
        if ignore_value is not None:
            ignore_value = parse_date(ignore_value)
    return field_value, ignore_value


def filter_by_fixed_find_all_kwargs(field_name: str, field_data: FixedFilterBy) -> dict[str, Any]:
    field_value, ignore_value = _maybe_transform_field_data(
        field_data.is_date, field_name, field_data.value, field_data.ignore_value
    )
    if field_data.is_date and (field_data.is_singularized or field_data.is_uncased):
        raise Exception(f"Date field {field_name} cannot be singularized or uncased.")
    if field_data.is_singularized and not field_data.is_uncased:
        raise Exception(
            f"Cased singularization of field ({field_name}) is not supported currently."
        )
    if field_value == ignore_value:
        return {}
    find_all_kwargs = {}
    if field_data.is_date:
        find_all_kwargs[f"{field_name}__ge"] = field_value.start_of("day")
        find_all_kwargs[f"{field_name}__le"] = field_value.end_of("day")
    elif field_data.is_singularized:  # should be before is_uncased.
        find_all_kwargs[f"{field_name}__is_singularized"] = field_value
    elif field_data.is_uncased:
        find_all_kwargs[f"{field_name}__is_uncased"] = field_value
    else:
        find_all_kwargs[field_name] = field_value
    return find_all_kwargs


def filter_by_has_find_all_kwargs(field_name: str, field_data: HasFilterBy) -> dict[str, Any]:
    field_value, ignore_value = _maybe_transform_field_data(
        False, field_name, field_data.value, field_data.ignore_value
    )
    if field_value == ignore_value:
        return {}
    if field_value not in [True, False]:
        raise Exception(f"Unexpected value for field_name {field_name}: {field_value}")
    find_all_kwargs = {}
    if field_value:
        find_all_kwargs[f"{field_name}__exists"] = field_value
    else:
        find_all_kwargs[f"{field_name}__not_exists"] = field_value
    return find_all_kwargs


def filter_by_membership_find_all_kwargs(
    field_name: str, field_data: MembershipFilterBy
) -> dict[str, Any]:
    field_value, ignore_value = _maybe_transform_field_data(
        field_data.is_date, field_name, field_data.value, field_data.ignore_value
    )
    if field_value == ignore_value:
        return {}
    find_all_kwargs = {f"{field_name}__has": field_value}
    return find_all_kwargs


def filter_by_range_find_all_kwargs(field_name: str, field_data: RangeFilterBy) -> dict[str, Any]:
    field_min, field_ignore_min = _maybe_transform_field_data(
        field_data.is_date, "min_" + field_name, field_data.min, field_data.ignore_min
    )
    field_max, field_ignore_max = _maybe_transform_field_data(
        field_data.is_date, "max_" + field_name, field_data.max, field_data.ignore_max
    )
    find_all_kwargs = {}
    if field_data.is_date:
        if field_min != field_ignore_min:
            find_all_kwargs[f"{field_name}__ge"] = field_min.start_of("day")
        if field_max != field_ignore_max:
            find_all_kwargs[f"{field_name}__le"] = field_max.end_of("day")
    if field_min == field_ignore_min and field_max == field_ignore_max:
        return {}
    if field_min == field_ignore_min:
        find_all_kwargs[f"{field_name}__le"] = (
            field_max.end_of("day") if isinstance(field_max, datetime) else field_max  # type: ignore
        )
    elif field_max == field_ignore_max:
        find_all_kwargs[f"{field_name}__ge"] = (
            field_min.start_of("day") if isinstance(field_min, datetime) else field_min  # type: ignore
        )
    else:
        find_all_kwargs[f"{field_name}__le"] = (
            field_max.end_of("day") if isinstance(field_max, datetime) else field_max  # type: ignore
        )
        find_all_kwargs[f"{field_name}__ge"] = (
            field_min.start_of("day") if isinstance(field_min, datetime) else field_min  # type: ignore
        )
    return find_all_kwargs


@validate_call
def sort_records(
    records: list[TSQLModel],
    sort_by: str | None = None,
    value_map: dict[Any, Any] | None = None,
    valid_fields: Sequence[str] | None = None,
) -> list[TSQLModel]:
    from appworld.apps.lib.apis.pagination import if_sort_by_is_invalid_raise_error

    if_sort_by_is_invalid_raise_error(sort_by, valid_fields)
    if not sort_by:
        return records
    if not sort_by.startswith("+") and not sort_by.startswith("-"):
        raise ValueError(f"Invalid sort_by value: {sort_by}.")
    if sort_by and (sort_by.startswith("+") or sort_by.startswith("-")):
        reverse = sort_by.startswith("-")
        sort_by = sort_by.lstrip("+").lstrip("-")
        records = sorted_plus(
            records,
            key=lambda record: record.field(sort_by),
            reverse=reverse,
            value_map=value_map,
        )
    return records
