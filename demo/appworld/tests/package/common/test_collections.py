import pytest

from appworld.common import random
from appworld.common.collections import (
    chunk_list,
    flatten,
    getter_plus,
    override_dict,
    sorted_plus,
    unique_by,
    unique_least_frequent,
    unique_max,
    unique_max_by,
    unique_min,
    unique_min_by,
    unique_most_frequent,
)


class TestCommonCollections:
    def test_unique_max_by(self) -> None:
        key = "key"
        values_to_dicts = lambda values: [{key: value} for value in values]
        assert unique_max_by(values_to_dicts([]), key) == (None, [])
        assert unique_max_by(values_to_dicts([1]), key) == (None, [])
        assert unique_max_by(values_to_dicts([1, 1]), key) == (None, [])
        assert unique_max_by(values_to_dicts([1, 2]), key) == ({key: 2}, [])
        assert unique_max_by(values_to_dicts([1, 1, 2]), key) == ({key: 2}, [])
        assert unique_max_by(values_to_dicts([1, 1, 2, 2]), key) == (
            {key: 2},
            [{key: 2}],
        )

    def test_unique_min_by(self) -> None:
        key = "key"
        values_to_dicts = lambda values: [{key: value} for value in values]
        assert unique_min_by(values_to_dicts([]), key) == (None, [])
        assert unique_min_by(values_to_dicts([1]), key) == (None, [])
        assert unique_min_by(values_to_dicts([1, 1]), key) == (None, [])
        assert unique_min_by(values_to_dicts([1, 2]), key) == ({key: 1}, [])
        assert unique_min_by(values_to_dicts([1, 2, 2]), key) == ({key: 1}, [])
        assert unique_min_by(values_to_dicts([1, 1, 2, 2]), key) == (
            {key: 1},
            [{key: 1}],
        )

    def test_unique_most_frequent(self) -> None:
        assert unique_most_frequent([]) == (None, [])
        assert unique_most_frequent([1]) == (None, [])
        assert unique_most_frequent([1, 1]) == (None, [])
        assert unique_most_frequent([1, 2]) == (None, [])
        assert unique_most_frequent([1, 2, 2]) == (2, [])
        assert unique_most_frequent([1, 1, 2, 2]) in ((1, [2]), (2, [1]))

    def test_unique_least_frequent(self) -> None:
        assert unique_least_frequent([]) == (None, [])
        assert unique_least_frequent([1]) == (None, [])
        assert unique_least_frequent([1, 1]) == (None, [])
        assert unique_least_frequent([1, 2]) == (None, [])
        assert unique_least_frequent([1, 2, 2]) == (1, [])
        assert unique_least_frequent([1, 1, 2, 2]) in ((1, [2]), (2, [1]))

    def test_unique_max(self) -> None:
        assert unique_max([]) is None
        assert unique_max([1, 1]) is None
        assert unique_max([1, 2]) == 2
        assert unique_max([1, 1, 2, 3, 3]) is None

    def test_unique_min(self) -> None:
        assert unique_min([]) is None
        assert unique_min([1, 1]) is None
        assert unique_min([1, 2]) == 1
        assert unique_min([1, 1, 2, 3, 3]) is None

    def test_unique_by(self) -> None:
        assert unique_by(
            [{"name": "A", "value": 1}, {"name": "B", "value": 2}, {"name": "A", "value": 3}],
            key="name",
            prefer_first=True,
        ) == [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
        assert unique_by(
            [{"name": "A", "value": 1}, {"name": "B", "value": 2}, {"name": "A", "value": 3}],
            key="name",
            prefer_first=False,
        ) == [{"name": "B", "value": 2}, {"name": "A", "value": 3}]
        assert unique_by(
            [{"name": "A", "value": 1}, {"name": "B", "value": 2}, {"name": "C", "value": 3}],
            key="name",
            prefer_first=True,
        ) == [{"name": "A", "value": 1}, {"name": "B", "value": 2}, {"name": "C", "value": 3}]
        assert unique_by(
            [{"name": "A", "value": 1}, {"name": "A", "value": 2}, {"name": "A", "value": 3}],
            key="name",
            prefer_first=False,
        ) == [{"name": "A", "value": 3}]
        assert unique_by(
            [{"name": "A", "value": 1}, None],
            key="name",
            skip_none=True,
        ) == [{"name": "A", "value": 1}]
        assert unique_by(
            [
                {"name": "A", "value": 5},
                {"name": "B", "value": 4},
                {"name": "C", "value": 3},
                {"name": "A", "value": 2},
                {"name": "B", "value": 1},
            ],
            key="name",
            prefer_first=True,
        ) == [
            {"name": "A", "value": 5},
            {"name": "B", "value": 4},
            {"name": "C", "value": 3},
        ]
        assert unique_by([], key="name", prefer_first=True) == []

    def test_sorted_plus(self) -> None:
        assert sorted_plus(random.shuffled([1, 2, 3, 4, 5])) == [1, 2, 3, 4, 5]
        assert sorted_plus(random.shuffled([1, 2, 3, None, None])) == [
            1,
            2,
            3,
            None,
            None,
        ]
        assert sorted_plus(random.shuffled([1, 2, 3, None, None]), reverse=True) == [
            3,
            2,
            1,
            None,
            None,
        ]
        assert sorted_plus(
            random.shuffled([1, 2, 3, None, None]), value_map={1: 3, 2: 2, 3: 1}
        ) == [3, 2, 1, None, None]
        assert sorted_plus(
            random.shuffled([1, 2, 3, None, None]), key=lambda x: x, value_map={1: 3, 2: 2, 3: 1}
        ) == [3, 2, 1, None, None]

    def test_flatten(self) -> None:
        assert flatten([]) == []
        assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
        assert flatten([[1, 2], [3, 4]], to="list") == [1, 2, 3, 4]
        assert flatten([(1, 2), (1, 2)], to="set") == {1, 2}
        assert flatten(((1, 2), (1, 2)), to="tuple") == (1, 2, 1, 2)

    def test_chunk_list(self) -> None:
        items = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            chunk_list(items, chunk_size=None, num_chunks=None)
        with pytest.raises(ValueError):
            chunk_list(items, chunk_size=2, num_chunks=2)
        assert chunk_list(items, chunk_size=1) == [[1], [2], [3], [4], [5]]
        assert chunk_list(items, chunk_size=2) == [[1, 2], [3, 4], [5]]
        assert chunk_list(items, chunk_size=3) == [[1, 2, 3], [4, 5]]
        assert chunk_list(items, chunk_size=4) == [[1, 2, 3, 4], [5]]
        assert chunk_list(items, chunk_size=5) == [[1, 2, 3, 4, 5]]
        with pytest.raises(ValueError):
            chunk_list(items, chunk_size=6)
        assert chunk_list(items, num_chunks=1) == [[1, 2, 3, 4, 5]]
        assert chunk_list(items, num_chunks=2) == [[1, 2], [3, 4, 5]]
        assert chunk_list(items, num_chunks=3) == [[1], [2], [3, 4, 5]]
        assert chunk_list(items, num_chunks=4) == [[1], [2], [3], [4, 5]]
        assert chunk_list(items, num_chunks=5) == [[1], [2], [3], [4], [5]]
        with pytest.raises(ValueError):
            chunk_list(items, num_chunks=6)
        assert chunk_list(items, num_chunks=1, balanced=True) == [[1, 2, 3, 4, 5]]
        assert chunk_list(items, num_chunks=2, balanced=True) == [[1, 2, 3], [4, 5]]
        assert chunk_list(items, num_chunks=3, balanced=True) == [[1, 2], [3, 4], [5]]
        assert chunk_list(items, num_chunks=4, balanced=True) == [[1, 2], [3], [4], [5]]
        assert chunk_list(items, num_chunks=5, balanced=True) == [
            [1],
            [2],
            [3],
            [4],
            [5],
        ]
        with pytest.raises(ValueError):
            chunk_list(items, num_chunks=6, balanced=True)
        items = list(range(11))  # 0..10
        unbalanced = chunk_list(items, num_chunks=4)  # -> [2,2,2,5] (last one is large)
        balanced = chunk_list(
            items, num_chunks=4, balanced=True
        )  # -> [3,3,3,2] (differ by 1 at most)
        assert [len(c) for c in unbalanced] == [2, 2, 2, 5]
        assert [len(c) for c in balanced] == [3, 3, 3, 2]
        assert [x for chunk in unbalanced for x in chunk] == items
        assert [x for chunk in balanced for x in chunk] == items

    def test_getter_plus(self) -> None:
        object_ = {"a": {"b": {"c": 1}}}
        assert getter_plus(object_, "a.b.c") == 1
        assert getter_plus(object_, "a_dot_b_dot_c") == 1
        assert getter_plus(object_, "a_dot_b.c") == 1
        object_ = {"a": [{"b": 1}, {"b": 2}]}
        assert getter_plus(object_, "a[0].b") == 1
        assert getter_plus(object_, "a[1].b") == 2
        object_ = {"a": {"b": [1, 2]}}
        assert getter_plus(object_, "a.b[0]") == 1
        assert getter_plus(object_, "a.b[1]") == 2
        object_ = {"a": [{"b": 1}, {"b": 2}]}
        assert getter_plus(object_, "a.b") == [1, 2]
        object_ = {"a": [{"b": {"c": [1, 2]}}, {"b": {"c": [3, 4]}}]}
        assert getter_plus(object_, "a.b.c") == [[1, 2], [3, 4]]
        assert getter_plus(object_, "a.b.c[0]") == [1, 3]
        assert getter_plus(object_, "a.b.c[1]") == [2, 4]
        object_ = [{"b": {"c": [1, 2]}}, {"b": {"c": [3, 4]}}]
        assert getter_plus(object_, "b.c[1]") == [2, 4]

    def test_override_dict(self):
        original = {"a": 1, "b": 2}
        updates = {"b": 3, "c": 4}
        expected = {"a": 1, "b": 3, "c": 4}
        override_dict(original, updates)
        assert original == expected
        # nested_update
        original = {"a": {"b": 1, "c": 2}}
        updates = {"a": {"b": 10}}
        expected = {"a": {"b": 10, "c": 2}}
        override_dict(original, updates)
        assert original == expected
        # add_new_nested_key
        original = {"a": {"b": 1}}
        updates = {"a": {"c": 2}}
        expected = {"a": {"b": 1, "c": 2}}
        override_dict(original, updates)
        assert original == expected
        # deeply_nested_update
        original = {"a": {"b": {"c": 1}}}
        updates = {"a": {"b": {"d": 2}}}
        expected = {"a": {"b": {"c": 1, "d": 2}}}
        override_dict(original, updates)
        assert original == expected
        # override_with_non_dict
        original = {"a": {"b": {"c": 1}}}
        updates = {"a": {"b": 5}}
        expected = {"a": {"b": 5}}
        override_dict(original, updates)
        assert original == expected
        # empty_updates
        original = {"a": 1}
        updates = {}
        expected = {"a": 1}
        override_dict(original, updates)
        assert original == expected
        # empty_original
        original = {}
        updates = {"a": {"b": 1}}
        expected = {"a": {"b": 1}}
        override_dict(original, updates)
        assert original == expected
