import copy
import operator
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast, overload

from appworld.common.types import Comparable, Number, TItem, TValue


def is_ascending(items: list[Comparable]) -> bool:
    if len(items) <= 1:
        return False
    return all(operator.le(items[index], items[index + 1]) for index in range(len(items) - 1))


def is_descending(items: list[Comparable]) -> bool:
    if len(items) <= 1:
        return False
    return all(operator.ge(items[index], items[index + 1]) for index in range(len(items) - 1))


def has_identical_sets(list_of_sets: list[set[Any]]) -> bool:
    # Checks if the list of sets has identical sets.
    for i in range(len(list_of_sets)):
        for j in range(i + 1, len(list_of_sets)):
            if list_of_sets[i] == list_of_sets[j]:
                return True
    return False


def are_sets_unique(sets: list[set[Any]]) -> bool:
    return len(set.intersection(*sets)) == 0


def are_disjoint(sets: list[set[Any] | list[Any] | tuple[Any, ...]]) -> bool:
    """
    Check if all the sets/lists/tuples are disjoint.
    """
    for index_1, set_1 in enumerate(sets):
        for index_2, set_2 in enumerate(sets):
            if index_1 == index_2:
                continue
            if not set(set_1).isdisjoint(set_2):
                return False
    return True


def copy_dict(
    dict: dict[str, TValue],
    keep: Iterable[str] | None = None,
    ignore: Iterable[str] | None = None,
) -> dict[str, TValue]:
    keep = keep or dict.keys()
    ignore = ignore or []
    return {key: value for key, value in dict.items() if key in keep and key not in ignore}


def override_dict(original_dict: dict[str, Any], dict_updates: dict[str, Any]) -> None:
    """Recursively and in-place update a nested dictionary with another nested dictionary."""
    for key, value in dict_updates.items():
        if (
            key in original_dict
            and isinstance(original_dict[key], dict)
            and isinstance(value, dict)
        ):
            override_dict(original_dict[key], value)
        else:
            original_dict[key] = value


def non_none(items: Iterable[TItem | None]) -> list[TItem]:
    return [item for item in items if item is not None]


def getter_plus(item: Any, key: str, allow_nested: bool = True, broadcast_list: bool = True) -> Any:
    """
    A general-purpose getter that works on nested paths (with . or _dot_ as delimiters),
    and works for both dictionary and object attributes. It also allows referencing
    list items with index, e.g. key="a[2]" will return the 3rd item of the list. It
    can be done arbitrarily anywhere in the path.
    If any of the intermediate result is a list, the the subsequent keys will be applied
    to all the items of the list, and the result will be a list of results.
    """
    directly_gettable = (isinstance(item, dict) and key in item) or hasattr(item, key)
    if not directly_gettable and ("." in key or "_dot_" in key) and allow_nested:
        for sub_key in re.split(r"\.|_dot_", key):
            item = getter_plus(item, sub_key)
        return item
    index: int | None = None
    if key.endswith("]"):
        index_str = key.rstrip("]").split("[", 1)[1]
        if not index_str.isnumeric():
            raise Exception(f"Invalid index in getting attribute for key {key}: {index_str}")
        index = int(index_str)
        key = key.split("[", 1)[0]
    input_was_list = isinstance(item, list)
    if not broadcast_list and input_was_list:
        raise Exception("Can't broadcast list input when broadcast_list is False.")
    items = [item] if not input_was_list else item
    result: list[Any] = []
    for item in items:
        item = item[key] if isinstance(item, dict) else getattr(item, key)
        if index is not None:
            item = item[index]
        result.append(item)
    if not input_was_list:
        result = result[0]
    return result


def sorted_plus(
    items: list[TItem],
    key: Callable[[TItem], Any] | None = None,
    reverse: bool = False,
    value_map: dict[Any, Any] | None = None,
) -> list[TItem]:
    # same as sorted except allows for values to be None, keeping them at the end.
    value_map = value_map or {}

    def sorting_function(item: TItem) -> Any:
        value = key(item) if key is not None else item
        if value in value_map:
            value = value_map[value]
        if reverse:
            return (value is not None, value)
        return (value is None, value)

    return sorted(items, key=sorting_function, reverse=reverse)


def sorted_by(items: list[TItem], key: str, reverse: bool = False) -> list[TItem]:
    return sorted(
        items,
        key=lambda item: getter_plus(item, key),
        reverse=reverse,
    )


def intesect_by(list_of_items: list[list[TItem]], key: str) -> list[TItem]:
    if not list_of_items:
        return []
    if len(list_of_items) == 1:
        return list_of_items[0]
    intersecting_key_values = set.intersection(
        *[{getter_plus(item, key) for item in items} for items in list_of_items]
    )
    return [item for item in list_of_items[0] if getter_plus(item, key) in intersecting_key_values]


def unique_by(
    items: list[TItem], key: str, skip_none: bool = False, prefer_first: bool = False
) -> list[TItem]:
    if skip_none:
        items = non_none(items)
    seen_keys: set[Any] = set()
    unique_items: list[TItem] = []
    iterable = reversed(items) if not prefer_first else items
    for item in iterable:
        if getter_plus(item, key) not in seen_keys:
            seen_keys.add(getter_plus(item, key))
            unique_items.append(item)
    if not prefer_first:
        unique_items.reverse()
    return unique_items


def tuple_of(items: list[TItem], key: str) -> tuple[Any]:
    return tuple([getter_plus(item, key) for item in items])


def list_of(items: list[TItem], key: str) -> list[Any]:
    return [getter_plus(item, key) for item in items]


def set_of(items: list[TItem], key: str) -> set[Any]:
    return {getter_plus(item, key) for item in items}


def dict_of(items: list[TItem], key: str, value: str) -> dict[Any, Any]:
    return {getter_plus(item, key): getter_plus(item, value) for item in items}


def dict_list_of(items: list[TItem], key: str, value: str) -> dict[Any, list[Any]]:
    result: dict[Any, list[Any]] = defaultdict(list)
    for item in items:
        result[getter_plus(item, key)].append(getter_plus(item, value))
    return result


def dict_set_of(items: list[TItem], key: str, value: str) -> dict[Any, set[Any]]:
    result: dict[Any, set[Any]] = defaultdict(set)
    for item in items:
        result[getter_plus(item, key)].add(getter_plus(item, value))
    return result


def unique_list_of(items: list[TItem], key: str) -> list[Any]:
    return unique([getter_plus(item, key) for item in items])


def lengths_of(items: Iterable[Iterable[TItem]], key: str) -> list[int]:
    return [len(getter_plus(item, key)) for item in items]


def max_by(items: list[TItem], key: str) -> TItem:
    return max(items, key=lambda item: getter_plus(item, key))


def min_by(items: list[TItem], key: str) -> TItem:
    return min(items, key=lambda item: getter_plus(item, key))


def unique_max_of(items: list[TItem], key: str) -> Number | None:
    # Simply composition of unique_max and list_of.
    return unique_max(list_of(items, key))


def unique_min_of(items: list[TItem], key: str) -> Number | None:
    # Simply composition of unique_min and list_of.
    return unique_min(list_of(items, key))


def unique_max_by(items: list[TItem], key: str) -> tuple[TItem | None, list[TItem]]:
    """
    Returns the item that has maximum value in the list according to the key.
    If there is <= 1 unique value in the list, returns None. If there are multiple
    candidates, returns one of them randomly along with the rest of the candidates
    in a list.
    See: tests/test_utils.py:test_unique_max_by
    """
    index_to_value: dict[int, Comparable] = {}
    for index, item in enumerate(items):
        value = cast(Comparable, getter_plus(item, key))
        index_to_value[index] = value
    values: set[Comparable] = set(index_to_value.values())
    if len(values) <= 1:
        return None, []
    highest_value = max(values)
    items_with_highest_value = [
        item for index, item in enumerate(items) if index_to_value[index] == highest_value
    ]
    sampled_item_with_highest_value = items_with_highest_value.pop(
        random.randrange(len(items_with_highest_value))
    )
    alternative_items_with_highest_value = items_with_highest_value
    return sampled_item_with_highest_value, alternative_items_with_highest_value


def unique_min_by(items: list[TItem], key: str) -> tuple[TItem | None, list[TItem]]:
    """
    Returns the item that has minimum value in the list according to the key.
    If there is <= 1 unique value in the list, returns None. If there are multiple
    candidates, returns one of them randomly along with the rest of the candidates
    in a list.
    See: tests/test_utils.py:test_unique_min_by
    """
    index_to_value: dict[int, Comparable] = {}
    for index, item in enumerate(items):
        value = cast(Comparable, getter_plus(item, key))
        index_to_value[index] = value
    values: set[Comparable] = set(index_to_value.values())
    if len(values) <= 1:
        return None, []
    lowest_value = min(values)
    items_with_lowest_value = [
        item for index, item in enumerate(items) if index_to_value[index] == lowest_value
    ]
    sampled_item_with_lowest_value = items_with_lowest_value.pop(
        random.randrange(len(items_with_lowest_value))
    )
    alternative_items_with_lowest_value = items_with_lowest_value
    return sampled_item_with_lowest_value, alternative_items_with_lowest_value


def unique_most_frequent(items: list[TItem]) -> tuple[TItem | None, list[TItem]]:
    """
    Returns the item that occurs most number of times in the list.
    If there is <= 1 unique items in the list, returns None. If there are multiple
    candidates, returns one of them randomly along with the rest of the candidates.
    in a list.
    See: tests/test_utils.py:test_unique_most_frequent
    """
    if len(unique(items)) <= 1:
        return None, []
    item_to_count = Counter(items)
    highest_frequency = item_to_count.most_common()[0][1]
    if highest_frequency <= 1:
        return None, []
    most_frequent_items = [
        item for item, count in item_to_count.items() if count == highest_frequency
    ]
    sampled_most_frequent_item = most_frequent_items.pop(
        random.randrange(len(most_frequent_items)) if len(most_frequent_items) > 1 else 0
    )
    alternative_most_frequent_items = most_frequent_items
    return sampled_most_frequent_item, alternative_most_frequent_items


def unique_least_frequent(items: list[TItem]) -> tuple[TItem | None, list[TItem]]:
    """
    Returns the item that occurs least number of times in the list.
    If there is <= 1 unique items in the list, returns None. If there are multiple
    candidates, returns one of them randomly along with the rest of the candidates.
    in a list.
    See: tests/test_utils.py:test_unique_least_frequent
    """
    if len(unique(items)) <= 1:
        return None, []
    item_to_count = Counter(items)
    most_frequency = item_to_count.most_common()[0][1]  # yes, it should be most.
    if most_frequency <= 1:
        return None, []
    lowest_frequency = item_to_count.most_common()[-1][1]
    least_frequent_items = [
        item for item, count in item_to_count.items() if count == lowest_frequency
    ]
    sampled_least_frequent_item = least_frequent_items.pop(
        random.randrange(len(least_frequent_items)) if len(least_frequent_items) > 1 else 0
    )
    alternative_least_frequent_items = least_frequent_items
    return sampled_least_frequent_item, alternative_least_frequent_items


def unique(items: list[TItem], skip_none: bool = False) -> list[TItem]:
    # NOTE: Do not use list(set(items)). It's changes everytime for
    # for some data types, e.g., for strings, even if random seed is set.
    # Put this in unstable.py
    # >> import random
    # >> random.seed(100)
    # >> print(list(set(["1", "2", "3"])))
    # and run it 3 times, you'll different results everytime.
    # If you're in the same file, you won't see different results.
    # The workaround is to set PYTHONHASHSEED before running python script
    # but it doesn't work if that env variable is set by os.environ.
    items = list(dict.fromkeys(items).keys())
    if skip_none:
        items = non_none(items)
    return items


def unique_max(items: Iterable[Number], **kwargs: Any) -> Number | None:
    """
    Same as max, but returns None if max can't be uniquely determined or there is no
    range to take max over. E.g.,:
    >> unique_max([]) => None
    >> unique_max([3]) => None
    >> unique_max([1, 1]) => None # there's no range of values to take max over
    >> unique_max([1, 2]) => 2
    >> unique_max([1, 2, 2, 3, 3]) => None # max can't be uniquely determined.
    """
    items = list(items)
    if len(items) <= 1:
        return None
    max_val = cast(Number, max(items, **kwargs))
    if items.count(max_val) > 1:
        return None
    return max_val


def unique_min(items: Iterable[Number], **kwargs: Any) -> Number | None:
    """
    Same as min, but returns None if min can't be uniquely determined or there is no
    range to take min over. E.g.,:
    >> unique_min([]) => None
    >> unique_min([1, 1]) => None # there's no range of values to take min over
    >> unique_min([1, 2]) => 1
    >> unique_min([1, 1, 2, 3, 3]) => None # min can't be uniquely determined.
    """
    items = list(items)
    if not items:
        return None
    min_val = cast(Number, min(items, **kwargs))
    if items.count(min_val) > 1:
        return None
    return min_val


def all_unique(items: Iterable[TItem], key: str | None = None) -> bool:
    if key is None:
        return len(set(items)) == len(list(items))
    return len({getter_plus(item, key) for item in items}) == len(list(items))


def dict_by(items: list[dict[str, Any]], key: str, remove_key: bool = False) -> dict[Any, Any]:
    result_dict: dict[Any, Any] = {}
    for item in items:
        outer_key = getter_plus(item, key)
        removable = isinstance(item, dict) and key in item
        if remove_key and removable:
            item = copy.deepcopy(item)
            item.pop(key)
        if remove_key and not removable:
            raise Exception(f"Can't pop key {key} from item {item}.")
        result_dict[outer_key] = item
    return result_dict


def dict_with_keys(dict_: dict[str, TItem], keys: list[str]) -> dict[str, TItem]:
    return {key: dict_[key] for key in keys}


def union(*sets_or_lists: Iterable[TItem]) -> set[TItem]:
    return set.union(*[set(set_or_list) for set_or_list in sets_or_lists])


def intersection(*sets_or_lists: Iterable[TItem]) -> set[TItem]:
    return set.intersection(*[set(set_or_list) for set_or_list in sets_or_lists])


@overload
def flatten(sequences: Sequence[Sequence[TItem]], to: Literal["list"] = "list") -> list[TItem]: ...


@overload
def flatten(sequences: Sequence[Sequence[TItem]], to: Literal["tuple"]) -> tuple[TItem, ...]: ...


@overload
def flatten(sequences: Sequence[Sequence[TItem]], to: Literal["set"]) -> set[TItem]: ...


def flatten(
    sequences: Sequence[Sequence[TItem]], to: Literal["list", "tuple", "set"] = "list"
) -> Any:
    if to == "list":
        return [item for sequence in sequences for item in sequence]
    elif to == "tuple":
        return tuple([item for sequence in sequences for item in sequence])
    elif to == "set":
        return {item for sequence in sequences for item in sequence}
    else:
        raise Exception(f"Invalid type in flatten: {type(sequences)}")


def field_values_occurring_n_times(
    instances: list[dict[str, Any]], field_name: str, n: int
) -> set[Any]:
    """
    Given a list of instances, return the set of field values that occur exactly n times.
    NOTE: This only works for non-mutable field values.
    """
    field_value_to_count: dict[Any, int] = Counter()
    for instance in instances:
        field_value_to_count[instance[field_name]] += 1
    return {field_value for field_value, count in field_value_to_count.items() if count == n}


def are_items_comparable(items: list[Any]) -> bool:
    if not items:
        return False
    first_item = items[0]
    for item in items[1:]:
        try:
            _ = first_item < item
        except TypeError:
            return False
    return True


def make_groups(items: list[TItem], min_group_size: int, max_group_size: int) -> list[list[TItem]]:
    """
    Given a list of items, make groups of items such that each group has at least
    min_group_size items and at most max_group_size items. Group size is randomized
    within these constraints.
    """
    original_num_items = len(items)
    # Shuffle the items to ensure randomness
    random.shuffle(items)
    # Calculate the number of groups and remaining items
    max_num_groups = len(items) // min_group_size
    remaining_items = len(items)
    # Initialize variables
    groups = []
    start_idx = 0
    # Iterate through the number of groups
    for _ in range(max_num_groups):
        # Calculate the actual group size
        if remaining_items >= min_group_size:
            group_size = random.randint(min_group_size, max_group_size)
            remaining_items -= group_size
        else:
            break
        # Add items to the current group
        end_idx = start_idx + group_size
        groups.append(items[start_idx:end_idx])
        start_idx = end_idx
    groups = [group for group in groups if group]
    # Add remaining items to the group that have space.
    for item in items[start_idx:]:
        for group in groups:
            if len(group) >= max_group_size:
                continue
            group.append(item)
            break
    for group in groups:
        assert min_group_size <= len(group) <= max_group_size
    assert original_num_items == sum([len(group) for group in groups])
    return groups


def chunk_list(
    input_list: list[TItem],
    chunk_size: int | None = None,
    num_chunks: int | None = None,
    balanced: bool = False,  # Newly added, so keeping it false by default to avoid breaking changes.
) -> list[list[TItem]]:
    if chunk_size is None and num_chunks is None:
        raise ValueError("Either chunk_size or num_chunks must be passed.")
    if chunk_size is not None and num_chunks is not None:
        raise ValueError("Only one of chunk_size or num_chunks can be passed.")
    if chunk_size is not None:
        if not chunk_size > 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}. Must be > 0.")
        if not chunk_size <= len(input_list):
            raise ValueError(
                f"Invalid chunk_size: {chunk_size}. Must be <= len(input_list): {len(input_list)}."
            )
        result = [
            input_list[index : index + chunk_size]
            for index in range(0, len(input_list), chunk_size)
        ]
    else:
        assert num_chunks is not None  # mypy
        if not num_chunks > 0:
            raise ValueError(f"Invalid num_chunks: {num_chunks}. Must be > 0.")
        if not num_chunks <= len(input_list):
            raise ValueError(
                f"Invalid num_chunks: {num_chunks}. Must be <= len(input_list): {len(input_list)}."
            )
        if balanced:
            n = len(input_list)
            base, extra = divmod(n, num_chunks)
            result = []
            start = 0
            for i in range(num_chunks):
                size = base + (1 if i < extra else 0)
                end = start + size
                result.append(input_list[start:end])
                start = end
        else:
            chunk_size = len(input_list) // num_chunks
            has_left_out = len(input_list) % num_chunks != 0
            result = [
                input_list[index : index + chunk_size]
                for index in range(0, len(input_list), chunk_size)
            ]
            if has_left_out:
                extra_items = flatten(result[num_chunks:])
                result_ = result[:num_chunks]
                result_[-1].extend(extra_items)
                result = result_
        assert len(result) == num_chunks
    return result


def chunk_and_return(
    input_list: list[TItem], num_chunks: int | None, chunk_index: int | None, balanced: bool = False
) -> list[TItem]:
    if not input_list:
        return []
    if num_chunks is None or chunk_index is None:
        return input_list
    num_chunks = int(num_chunks)
    chunk_index = int(chunk_index)
    if num_chunks <= 0:
        raise ValueError(f"Invalid num_chunks: {num_chunks}. Must be > 0.")
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(
            f"Invalid chunk index: {chunk_index}. Must be between 0 and {num_chunks - 1}."
        )
    chunks = chunk_list(input_list, num_chunks=num_chunks, balanced=balanced)
    return chunks[chunk_index]


def rolling_get(
    items: Sequence[TItem], index_or_position: int, zero_indexed: bool = False
) -> TItem:
    index = index_or_position if zero_indexed else index_or_position - 1
    return items[index % len(items)]


def diff_lists(list_1: list[Any], list_2: list[Any]) -> list[Any]:
    return [item for item in list_1 if item not in list_2]


def diff_dicts(
    dicts_1: list[dict[str, Any]],
    dicts_2: list[dict[str, Any]],
    ignore: Iterable[str] = (),
) -> list[dict[str, Any]]:
    # Diff dicts_1 - dicts_2
    stripped_dicts_1 = copy.deepcopy(dicts_1)
    stripped_dicts_2 = copy.deepcopy(dicts_2)
    for dict_ in stripped_dicts_1 + stripped_dicts_2:
        for key in ignore:
            dict_.pop(key, None)

    selected_dicts_1 = []
    for dict_1, stripped_dict_1 in zip(dicts_1, stripped_dicts_1, strict=True):
        if stripped_dict_1 not in stripped_dicts_2:
            selected_dicts_1.append(dict_1)

    return selected_dicts_1


def intersect_lists(*list_of_items: list[TItem]) -> list[TItem]:
    if not list_of_items:
        return []
    intersection_ = list_of_items[0]
    for sublist in list_of_items[1:]:
        intersection_ = [item for item in intersection_ if item in sublist]
    return intersection_


def subtract_lists(items_1: list[TItem], items_2: list[TItem]) -> list[TItem]:
    return [item for item in items_1 if item not in items_2]


def list_comparison_report(name_1: str, items_1: list[Any], name_2: str, items_2: list[Any]) -> str:
    items_1 = sorted(items_1)
    items_2 = sorted(items_2)
    items_1_not_in_2 = [item for item in items_1 if item not in items_2]
    items_2_not_in_1 = [item for item in items_2 if item not in items_1]
    report = "\n".join(
        [
            f"In {name_1} ; Not in {name_2} => {items_1_not_in_2}",
            f"In {name_2} ; Not in {name_1} => {items_2_not_in_1}",
        ]
    )
    return report
