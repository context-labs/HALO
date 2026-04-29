from appworld.common.finders import find_all


class TestCommonFinders:
    def test_find_all(self) -> None:
        # TODO: Add tests for some of the other operators (not necessary though as it's tested in cls.find_all)
        datum_1 = {
            "a": 1,
            "b": 2,
            "c": {"a": 3, "b": 4},
        }
        datum_2 = {
            "a": 7,
            "b": 8,
            "c": {"a": 9, "b": 10},
        }
        data = [datum_1, datum_2]
        assert find_all(data, a=1) == [datum_1]
        assert find_all(data, a=7) == [datum_2]
        assert find_all(data, c_dot_b=10) == [datum_2]
