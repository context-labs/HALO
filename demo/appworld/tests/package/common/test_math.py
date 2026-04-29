from appworld.common.math import range_plus


class TestCommonMath:
    def test_range_plus(self) -> None:
        assert list(range_plus(1, 3)) == [1, 2, 3]
        assert list(range_plus(1, 3, 0.5)) == [1, 1.5, 2, 2.5, 3]
        assert list(range_plus(3, 1, -1)) == [3, 2, 1]
        assert list(range_plus(3, 1, -0.5)) == [3, 2.5, 2, 1.5, 1]
        assert range_plus(23, 79, 10, round_ends_to=10) == [30, 40, 50, 60, 70]
        assert range_plus(23, 24, 10, round_ends_to=10) == []
