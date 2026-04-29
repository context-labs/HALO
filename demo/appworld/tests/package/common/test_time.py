import sys
import time as _time

import pytest

from appworld.common.time import Timer, freeze_time, timeout_call


class TestCommonTime:
    def test_timeout_call(self) -> None:
        if sys.platform == "win32":
            pytest.skip("timeout_call is not supported on Windows.")

        def function() -> int:
            _time.sleep(2)
            return 1

        assert timeout_call(function, timeout_seconds=3) == 1
        with pytest.raises(Exception) as exception:
            timeout_call(function, timeout_seconds=1)
        assert str(exception.value) == "Function function execution timed out after 1 seconds."

    def test_timer(self) -> None:
        if sys.platform == "win32":
            pytest.skip("Timer is not supported on Windows.")

        timer = Timer(start=True)
        _time.sleep(0.3)
        time_taken = timer.stop()
        assert time_taken >= 0.3
        with freeze_time("2020-10-06"):
            timer = Timer(start=True, bypass_freezegun=False)
            _time.sleep(0.3)
            time_taken = timer.stop()
            assert time_taken == 0
            timer = Timer(start=True, bypass_freezegun=True)
            _time.sleep(0.3)
            time_taken = timer.stop()
            assert time_taken >= 0
