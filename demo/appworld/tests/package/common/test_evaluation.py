import pytest

from appworld.common.datetime import DateTime
from appworld.common.evaluation import answer_to_text, assert_plus


class TestCommonEvaluation:
    def test_answer_to_text(self) -> None:
        assert answer_to_text(12.34) == "12.34"
        assert answer_to_text(12) == "12"
        assert answer_to_text("12.34") == "12.34"
        assert answer_to_text("12") == "12"
        assert answer_to_text([12.34, 56.78]) == "12.34 | 56.78"
        assert answer_to_text([12, 56.78]) == "12 | 56.78"
        assert answer_to_text([12, "56.78"]) == "12 | 56.78"
        assert answer_to_text([12, "56"]) == "12 | 56"
        assert answer_to_text([12, "56", 78]) == "12 | 56 | 78"
        assert answer_to_text([12, "56", 78][::-1]) == "12 | 56 | 78"
        assert answer_to_text("12.348") == "12.35"
        assert answer_to_text("twelve") == "12"
        assert answer_to_text("  one   thousand  two  hundred and thirty four ") == "1234"
        assert answer_to_text("  one   thousand  two  hundred thirty four ") == "1234"
        assert answer_to_text(" ' HeLlo   WORLD ' ") == "hello world"
        assert answer_to_text(' " HeLlo   WORLD " ') == "hello world"
        assert answer_to_text(None) == "null"
        assert answer_to_text("") == "null"

    def test_assert_plus(self) -> None:
        # Test all individual conditions:
        assert_plus(1, "==", 1)
        with pytest.raises(AssertionError):
            assert_plus(1, "==", 2)
        assert_plus(1, "!=", 2)
        with pytest.raises(AssertionError):
            assert_plus(1, "!=", 1)
        assert_plus(1, ">", 0)
        with pytest.raises(AssertionError):
            assert_plus(1, ">", 1)
        assert_plus(1, ">=", 1)
        assert_plus(1, ">=", 0)
        with pytest.raises(AssertionError):
            assert_plus(1, ">=", 2)
        assert_plus(1, "<", 2)
        with pytest.raises(AssertionError):
            assert_plus(1, "<", 1)
        assert_plus(1, "<=", 1)
        assert_plus(1, "<=", 2)
        with pytest.raises(AssertionError):
            assert_plus(1, "<=", 0)
        assert_plus(1, "is", 1)
        with pytest.raises(AssertionError):
            assert_plus(1, "is", 2)
        assert_plus(1, "is not", 2)
        with pytest.raises(AssertionError):
            assert_plus(1, "is not", 1)
        assert_plus(1, "is_truthy")
        with pytest.raises(AssertionError):
            assert_plus(0, "is_truthy")
        assert_plus(0, "is_falsy")
        with pytest.raises(AssertionError):
            assert_plus(1, "is_falsy")
        assert_plus(1, "in", [1, 2])
        with pytest.raises(AssertionError):
            assert_plus(1, "in", [2, 3])
        assert_plus(1, "not in", [2, 3])
        # Test all individual conditions with some additional args.
        assert_plus(1.0, "==", 1, tolerance=0.11)
        assert_plus(1.0, "!=", 2, tolerance=0.11)
        assert_plus({"1": [1.0, 1.0]}, "==", {"1": [1.5, 0.5]}, tolerance=1)
        assert_plus({1: [1.0, 1.0]}, "==", {1: [1.5, 0.5]}, tolerance=1, freeze_keys=True)
        with pytest.raises(AssertionError):
            assert_plus(1.0, "!=", 1, tolerance=0.11)
            assert_plus(1.0, "==", 2, tolerance=0.11)
        assert_plus([1.0, 1.0], "==", [1.5, 0.5], tolerance=1)
        assert_plus({"key": [1.0, 1.0]}, "==", {"key": [1.5, 0.5]}, tolerance=1)
        assert_plus("Apple", "==", "APPle", ignore_case=True)
        assert_plus("Apple", "!=", "Banana", ignore_case=True)
        assert_plus("Apple", "in", ["apple", "banana"], ignore_case=True)
        assert_plus(" apple ", "==", "Apple   ", strip=True, ignore_case=True)
        assert_plus(["Apple"], "==", ["APPle"], ignore_case=True)
        assert_plus({"Apple"}, "==", {"APPle"}, ignore_case=True)
        assert_plus([1, 2, 3], "==", [1, 3, 2], ignore_order=True)
        assert_plus([1, 2, 3], "!=", [1, 2, 4], ignore_order=True)
        assert_plus({"key": [1, 2, 3]}, "==", {"key": [1, 3, 2]}, ignore_order=True)
        assert_plus([[1, 2], [3, 4]], "==", [[4, 3], [2, 1]], ignore_order=True)
        assert_plus(2.358, "==", 2.359, round_to=2)
        assert_plus(" app  \nle", "==", "app le", merge_white_space=True)
        with pytest.raises(AssertionError):
            assert_plus("app  le", "==", "apple", merge_white_space=True)
        assert_plus("app  le", "==", "apple", ignore_white_space=True)
        assert_plus("APp   le", "==", "app le", normalize_text=True)
        assert_plus("apple", "==", "apples", singularize=True)
        assert_plus({"key": "APp le"}, "==", {"key": "app le"}, normalize_text=True)
        assert_plus({" kEy": "APp le"}, "==", {"key": "app le"}, normalize_text=True)
        assert_plus(["APp le", " Ban Ana "], "==", ["app le", "ban ana"], normalize_text=True)
        assert_plus(["app  le"], "==", ["apple"], ignore_white_space=True)
        assert_plus({1: ["app  le"]}, "==", {1: ["apple"]}, ignore_white_space=True)
        assert_plus(
            [["1", "2"], ["3", "4"]],
            "==",
            [[" 4 ", " 3 "], [" 2 ", "1"]],
            ignore_order=True,
            normalize_text=True,
        )
        assert_plus([{"apple"}, {"apple"}], "all <=", {"apple"})  # subset
        with pytest.raises(AssertionError):
            assert_plus([{"apple"}, {"banana"}], "all <=", {"apple"})
        assert_plus([{"apple"}, {"banana"}], "any <=", {"apple"})
        # Test with DateTime round_to:
        datetime_ = DateTime.now().set(second=0, microsecond=0)
        assert_plus(datetime_, "==", datetime_.add(seconds=1), round_to="minute")
        assert_plus(datetime_, "==", datetime_.subtract(seconds=1), round_to="minute")
        datetime_ = DateTime.now().set(minute=0, second=0, microsecond=0)
        assert_plus(datetime_, "==", datetime_.add(minutes=1), round_to="hour")
        assert_plus(datetime_, "==", datetime_.subtract(minutes=1), round_to="hour")
        datetime_ = DateTime.now().set(hour=0, minute=0, second=0, microsecond=0)
        assert_plus(datetime_, "==", datetime_.add(hours=1), round_to="day")
        assert_plus(datetime_, "==", datetime_.subtract(hours=1), round_to="day")
        assert_plus(
            {"key": datetime_},
            "==",
            {"key": datetime_.add(seconds=1)},
            round_to="minute",
        )
        # Test some with left operand all/any:
        assert_plus([1, 2], "all >=", 1)
        assert_plus([1, 2], "any >=", 2)
        # Test some with right operand all/any:
        assert_plus(2, ">= all", [1, 2])
        assert_plus(1, ">= any", [1, 2])
        # # Test some with both left and right operand all/any:
        assert_plus({6, 5, 4}, "all > all", {2, 3})
        assert_plus({6, 5, 4}, "all > any", {2, 5})
        assert_plus({2, 5}, "any > all", {4, 3, 2})
        assert_plus({1, 0, 3}, "any > any", {1, 0})
        # Test left/right operand all with empty list (non default semantics):
        with pytest.raises(AssertionError):
            assert_plus([], "all >=", 1)
        with pytest.raises(AssertionError):
            assert_plus(1, ">= all", [])
        with pytest.raises(AssertionError):
            assert_plus([], "all > all", [])
        assert_plus([], "all >=", 1, allow_left_empty=True)
        assert_plus(1, ">= all", [], allow_right_empty=True)
        assert_plus([], "all > all", [], allow_left_empty=True, allow_right_empty=True)
        with pytest.raises(ValueError):
            assert_plus(1, "==", 1, allow_right_empty=True)
            assert_plus(1, "==", 1, allow_left_empty=True)
        assert_plus([1, 2], "all >=", 1, allow_left_empty=True)
        assert_plus(2, ">= all", [1, 2], allow_right_empty=True)
        with pytest.raises(AssertionError):
            assert_plus(["a", "A"], "all ==", "a")
        assert_plus(["a", "A"], "all ==", "a", ignore_case=True)
        assert_plus(["a", " A "], "all ==", "a", normalize_text=True)
