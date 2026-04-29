import operator
import re
from collections.abc import Callable
from typing import Any, cast

from pendulum.datetime import DateTime as _DateTime
from text_to_num import text2num

from appworld.common.collections import diff_lists
from appworld.common.constants import NOT_PASSED
from appworld.common.errors import MaybeSuppressErrors
from appworld.common.types import AnswerType


def answer_to_text(answer: AnswerType) -> str:
    if isinstance(answer, str) and not answer.strip():
        return "null"
    if answer is None:
        return "null"
    if isinstance(answer, list | tuple):
        return " | ".join(sorted([answer_to_text(v) for v in answer]))
    if isinstance(answer, str):
        answer = answer.strip()
    if isinstance(answer, str) and (
        (answer.startswith('"') and answer.endswith('"'))
        or (answer.startswith("'") and answer.endswith("'"))
    ):
        answer = answer[1:-1]
    if isinstance(answer, str):
        try:
            answer = text2num(answer, "en")
        except ValueError:
            pass
    try:
        num = float(cast(Any, answer))
        if num.is_integer():
            answer = int(num)
        else:
            answer = round(num, 2)
    except (ValueError, TypeError):
        pass
    answer_text = str(answer).strip().lower()
    answer_text = " ".join(answer_text.split())
    return answer_text


def do_answers_match(predicted_answer: AnswerType, ground_truth_answer: AnswerType) -> bool:
    if predicted_answer == ground_truth_answer:
        return True
    if answer_to_text(predicted_answer) == answer_to_text(ground_truth_answer):
        return True
    return False


def assert_answers_match(predicted_answer: AnswerType, ground_truth_answer: AnswerType) -> None:
    assert_plus(answer_to_text(predicted_answer), "==", answer_to_text(ground_truth_answer))


def assert_plus(
    left: Any,
    condition: str = "is_truthy",
    right: Any = NOT_PASSED,
    message: str | None = None,
    normalize: Callable[[Any], Any] = lambda e: e,
    tolerance: float | None = None,
    ignore_case: bool = False,
    strip: bool = False,
    merge_white_space: bool = False,  # TODO: rename to merge_white_space
    ignore_white_space: bool = False,  # remove all whitespace.  # TODO: rename to ignore_white_space
    normalize_text: bool = False,  # shorthand for ignore_case, strip, merge_white_space
    singularize: bool = False,
    round_to: int | str | None = None,
    allow_left_empty: (bool | None) = None,  # only applicable if left modifier of "all" is given.
    allow_right_empty: (bool | None) = None,  # only applicable if right modifier of "all" is given.
    freeze_keys: bool = False,  # if True, transformations will not be applied to dict keys.
    ignore_order: bool = False,  # if True, order of list/tuple/set will be ignored, i.e., it'll be sorted.
) -> None:
    """
    Default assert with additional features:
    1. better messages out-of-the-box:
       Shows what left and right values are. Can be overridden by passing message explicitly.
    2. allows normalizing the operands by some function before comparing them.
    3. allows using all/any modifier on left/right operand.
       E.g., [1, 3, 4] "all >" 0 will assert that all element is greater than 0.
       and [5, 6, 7] "all > all" {2, 4} will assert all element of left is greater than all element of right.
       You can replace this with a for loop. But this is mainly for test.case(..) where
       all call to test.case(..) is of a unit "weight". Looping over test.case(..) can give
       unreasonable weight to the test case effectively.
       TODO: This functionality belongs to test.case(..). Move it there.
    Condition can be: =, ==, >, <, >=, <=, !=, in, not in, is, is not, is_truthy, is_falsy
    """
    import pytest  # expensive import so its kept local.

    if normalize_text:
        strip = ignore_case = merge_white_space = True
    modifiers = ["all", "any"]
    left_modifier: str | None = None
    right_modifier: str | None = None
    for modifier in modifiers:
        dict_keys: type = type({}.keys())
        dict_values: type = type({}.values())
        if condition.startswith(f"{modifier} "):
            assert isinstance(left, list | tuple | set | dict_keys | dict_values), (
                f"If {modifier} is passed on left operand of assert_plus, "
                "it must be an iterable (list/tuple/set/dict_keys/dict_values). "
                f"Found type: {type(left)}"
            )
            left_modifier = modifier
            if modifier == "all" and not allow_left_empty:
                assert left, "The left modifier is 'all' with allow_left_empty=False, but left operand is empty."
        if condition.endswith(f" {modifier}"):
            assert isinstance(right, list | tuple | set | dict_keys | dict_values), (
                f"If {modifier} is passed on right operand of assert_plus, "
                "it must be an iterable (list/tuple/set/dict_keys/dict_values). "
                f"Found type: {type(right)}"
            )
            right_modifier = modifier
            if modifier == "all" and not allow_right_empty:
                assert right, "The right modifier is 'all' with allow_right_empty=False, but right operand is empty."
    if left_modifier != "all" and allow_left_empty is not None:
        raise ValueError("allow_left_empty is only applicable if left modifier of 'all' is given.")
    if right_modifier != "all" and allow_right_empty is not None:
        raise ValueError(
            "allow_right_empty is only applicable if right modifier of 'all' is given."
        )
    if left_modifier:
        condition = condition.removeprefix(left_modifier).strip()
        pass_count = 0
        exceptions: list[BaseException] = []
        for left_ in left:
            with MaybeSuppressErrors(left_modifier == "any") as suppressor:
                assert_plus(
                    left_,
                    condition=condition,
                    right=right,
                    message=message,
                    normalize=normalize,
                    tolerance=tolerance,
                    ignore_case=ignore_case,
                    strip=strip,
                    merge_white_space=merge_white_space,
                    ignore_white_space=ignore_white_space,
                    normalize_text=normalize_text,
                    singularize=singularize,
                    round_to=round_to,
                    freeze_keys=freeze_keys,
                    ignore_order=ignore_order,
                )
            if not suppressor.raised_error:
                pass_count += 1
            else:
                assert suppressor.exception is not None
                exceptions.append(suppressor.exception)
        if left_modifier == "any" and pass_count == 0:
            raise exceptions[0]
        if left_modifier == "all" and pass_count != len(left):
            raise exceptions[0]
        return
    elif right_modifier:
        condition = condition.removesuffix(right_modifier).strip()
        pass_count = 0
        exceptions_right: list[BaseException] = []
        for right_ in right:
            with MaybeSuppressErrors(right_modifier == "any") as suppressor:
                assert_plus(
                    left,
                    condition=condition,
                    right=right_,
                    message=message,
                    normalize=normalize,
                    tolerance=tolerance,
                    ignore_case=ignore_case,
                    strip=strip,
                    merge_white_space=merge_white_space,
                    ignore_white_space=ignore_white_space,
                    normalize_text=normalize_text,
                    singularize=singularize,
                    round_to=round_to,
                    freeze_keys=freeze_keys,
                    ignore_order=ignore_order,
                )
            if not suppressor.raised_error:
                pass_count += 1
            else:
                assert suppressor.exception is not None
                exceptions_right.append(suppressor.exception)
        if right_modifier == "any" and pass_count == 0:
            raise exceptions_right[0]
        if right_modifier == "all" and pass_count != len(right):
            raise exceptions_right[0]
        return

    valid_conditions: dict[str, Callable[..., Any]] = {
        "=": operator.eq,
        "==": operator.eq,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "!=": operator.ne,
        "in": lambda x, y: operator.contains(y, x),
        "not in": lambda x, y: not operator.contains(y, x),
        "is": operator.is_,
        "is not": operator.is_not,
        "is_truthy": operator.is_,  # shorthand for bool(left) is True
        "is_falsy": operator.is_not,  # shorthand for bool(left) is False
    }
    if condition not in list(valid_conditions.keys()):
        # yes, list(..keys()) is required if condtion is a set
        raise ValueError(
            f"Invalid condition: {condition}. Valid ones are: {list(valid_conditions.keys())}"
        )

    original_left_repr = repr(left)
    original_right_repr = repr(right)

    if condition in ["is_truthy", "is_falsy"]:
        assert right == NOT_PASSED, "Can't use 'is_truthy' or 'is_falsy' with a right operand."
    elif right == NOT_PASSED:
        assert condition in [
            "is_truthy",
            "is_falsy",
        ], "If right operand is not passed, condition must be 'is_truthy' or 'is_falsy'."

    if condition in ["is_truthy", "is_falsy"]:
        left = bool(left)
        right = True

    comparison_operator = valid_conditions[condition]

    def apply_recursively(
        item: Any, function: Callable[..., Any], freeze_keys: bool = False
    ) -> Any:
        if isinstance(item, list):
            return [apply_recursively(sub_item, function, freeze_keys) for sub_item in item]
        elif isinstance(item, tuple):
            return tuple(apply_recursively(sub_item, function, freeze_keys) for sub_item in item)
        elif isinstance(item, set):
            return {apply_recursively(sub_item, function, freeze_keys) for sub_item in item}
        elif isinstance(item, dict):
            for key in list(item.keys()):
                value = item.pop(key)
                if not freeze_keys:
                    key = apply_recursively(key, function, freeze_keys)
                item[key] = apply_recursively(value, function, freeze_keys)
            return item
        else:
            return function(item)

    # normalize should not be applied element-wise.
    left = normalize(left)
    right = normalize(right)

    if tolerance is not None:

        def approximate(item: Any) -> Any:
            return pytest.approx(item, abs=tolerance) if isinstance(item, int | float) else item

        left = apply_recursively(left, approximate, freeze_keys)
        right = apply_recursively(right, approximate, freeze_keys)

    if round_to is not None:
        if isinstance(round_to, str) and round_to not in (
            "second",
            "minute",
            "hour",
            "day",
        ):
            raise ValueError(
                f"Invalid value for round_to: {round_to}. "
                "It should either be int or seconds, minutes, hours"
            )

        def round_(item: Any) -> Any:
            if isinstance(item, float | int) and isinstance(round_to, int):
                return round(item, round_to)
            elif isinstance(item, _DateTime) and round_to in (
                "second",
                "minute",
                "hour",
                "day",
            ):
                return cast(Any, item).round(round_to)
            return item

        left = apply_recursively(left, round_, freeze_keys)
        right = apply_recursively(right, round_, freeze_keys)

    if ignore_case:

        def lowercase(item: Any) -> Any:
            return item.lower() if isinstance(item, str) else item

        left = apply_recursively(left, lowercase, freeze_keys)
        right = apply_recursively(right, lowercase, freeze_keys)

    if strip:

        def strip_(item: Any) -> Any:
            return item.strip() if isinstance(item, str) else item

        left = apply_recursively(left, strip_, freeze_keys)
        right = apply_recursively(right, strip_, freeze_keys)

    if merge_white_space:

        def merge_white_space_fn(item: Any) -> Any:
            return re.sub(r"\s+", " ", item).strip() if isinstance(item, str) else item

        left = apply_recursively(left, merge_white_space_fn, freeze_keys)
        right = apply_recursively(right, merge_white_space_fn, freeze_keys)

    if ignore_white_space:

        def remove_whitespace(item: Any) -> Any:
            return re.sub(r"\s+", "", item) if isinstance(item, str) else item

        left = apply_recursively(left, remove_whitespace, freeze_keys)
        right = apply_recursively(right, remove_whitespace, freeze_keys)

    if singularize:

        def singularize_(item: Any) -> Any:
            from inflection import singularize

            return singularize(item) if isinstance(item, str) else item

        left = apply_recursively(left, singularize_, freeze_keys)
        right = apply_recursively(right, singularize_, freeze_keys)

    if ignore_order:

        def sort_(item: Any) -> Any:
            if isinstance(item, list):
                return sorted([sort_(x) for x in item])
            elif isinstance(item, tuple):
                return tuple(sorted([sort_(x) for x in item]))
            elif isinstance(item, dict):
                return {key: sort_(value) for key, value in item.items()}
            return item

        left = sort_(left)
        left = apply_recursively(left, sort_, freeze_keys)
        right = sort_(right)
        right = apply_recursively(right, sort_, freeze_keys)

    if comparison_operator(left, right):
        return

    if message:
        raise AssertionError(message)

    left_repr = repr(left)
    right_repr = repr(right)
    delimiter = "\n" if len(left_repr + right_repr) > 70 else " "
    message = delimiter.join(["", left_repr, condition, right_repr])

    for type_ in [set, tuple, list]:
        if condition == "==" and isinstance(left, type_) and isinstance(right, type_):
            in_left_but_not_right = diff_lists(left, right)
            in_right_but_not_left = diff_lists(right, left)
            if in_left_but_not_right:
                message += f"\n\nIn left but not right:\n{in_left_but_not_right}"
            if in_right_but_not_left:
                message += f"\n\nIn right but not left:\n{in_right_but_not_left}"

    if original_left_repr != left_repr or original_right_repr != right_repr:
        message += f"\n\nOriginal values:\n{delimiter.join([original_left_repr, condition, original_right_repr])}"

    raise AssertionError(message)
