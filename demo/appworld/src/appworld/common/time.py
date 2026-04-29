import asyncio
import os
import signal
import sys
from collections.abc import Callable
from contextlib import ContextDecorator
from datetime import datetime
from time import perf_counter
from typing import Any, Self, TypeVar, cast
from typing import get_args as get_type_args  # noqa: F401 (better name)

from freezegun import freeze_time as _freeze_time
from freezegun.api import _freeze_time as _freeze_time_class
from pendulum.datetime import DateTime as _DateTime


class Timer:
    def __init__(
        self, bypass_freezegun: bool = False, message: str | None = None, start: bool = False
    ) -> None:
        if not bypass_freezegun:
            self.get_time = perf_counter
        else:
            if os.name == "posix":  # linux/mac
                import uvloop

                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            elif os.name == "nt":  # windows
                import winloop

                print(
                    "WARNING: Note that Timer has not been tested with Windows "
                    "because of incompatible event loop library. Please report any issues."
                )
                asyncio.set_event_loop_policy(winloop.WinLoopPolicy())
            else:
                raise Exception(
                    f"The Timer is unsupported for OS: {os.name}. Only linux/mac/windows are supported."
                )
            self.get_time = _freezegun_bypassed_time
        self.message = message
        if start:
            self.start()

    def start(self) -> float:
        self.start_time = self.get_time()
        return self.start_time

    def stop(self) -> float:
        if self.start_time is None:
            raise Exception("Timer has not been started. Use start() to start the timer.")
        self.end_time = self.get_time()
        return float(self.end_time - self.start_time)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, type, value, traceback) -> None:  # type: ignore
        elapsed = self.stop()
        print(f"{self.message} took {elapsed:.3f} seconds")


class TimeoutError(Exception):
    pass


ReturnT = TypeVar("ReturnT")


def timeout_call(
    function: Callable[..., ReturnT],
    timeout_seconds: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> ReturnT:
    if timeout_seconds is None:
        return function(*args, **kwargs)
    timeout_seconds = int(timeout_seconds)

    if sys.platform == "win32":
        raise Exception("timeout_call is not supported on Windows.")

    function_name = getattr(function, "__name__", repr(function))
    timeout_message = (
        f"Function {function_name} execution timed out after {timeout_seconds} seconds."
    )

    def timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(timeout_message)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = function(*args, **kwargs)
    except TimeoutError as exception:
        raise Exception(timeout_message) from exception
    finally:
        signal.alarm(0)
    return result


class freeze_time(ContextDecorator):  # noqa: N801
    def __init__(self, system_datetime: datetime | None = None):
        if isinstance(system_datetime, _DateTime):
            # https://github.com/pallets/flask/issues/4738
            system_datetime = cast(Any, system_datetime).to_datetime()
        self.system_datetime: datetime | None = system_datetime
        self._freezer: _freeze_time_class | None = None

    def start(self) -> Self:
        self._freezer = _freeze_time(self.system_datetime)
        self._freezer.start()
        return self

    def stop(self) -> None:
        if self._freezer is not None:
            self._freezer.stop()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: Any | None, exc_val: Any | None, exc_tb: Any | None) -> None:
        self.stop()


async def _time_coroutine() -> float:
    return asyncio.get_event_loop().time()


def freezegun_bypassed_datetime() -> datetime:
    import ctypes
    import ctypes.util
    import ctypes.wintypes as wt

    if not sys.platform.startswith("win"):
        clock_real_time = 0

        class Timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]  # noqa: RUF012

        librt_path = ctypes.util.find_library("rt") or ctypes.util.find_library("c")
        if not librt_path:
            raise RuntimeError("Cannot find libc/librt for clock_gettime")
        librt = ctypes.CDLL(librt_path, use_errno=True)
        ts = Timespec()
        if librt.clock_gettime(clock_real_time, ctypes.byref(ts)) != 0:
            e = ctypes.get_errno()
            raise OSError(e, "clock_gettime failed")
        return datetime.fromtimestamp(ts.tv_sec + ts.tv_nsec / 1e9)
    else:
        GetSystemTimePreciseAsFileTime = ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime  # noqa: N806
        GetSystemTimePreciseAsFileTime.argtypes = [ctypes.POINTER(wt.FILETIME)]
        GetSystemTimePreciseAsFileTime.restype = None
        ft = wt.FILETIME()
        GetSystemTimePreciseAsFileTime(ctypes.byref(ft))
        t100ns = (ft.dwHighDateTime << 32) + ft.dwLowDateTime
        unix_100ns = t100ns - 116444736000000000
        sec, nsec = divmod(unix_100ns, 10_000_000)
        return datetime.fromtimestamp(sec + nsec / 10_000_000)


def _freezegun_bypassed_time() -> float:
    # 1. This bypass hack assumes https://github.com/spulec/freezegun/issues/529 is not fixed.
    # 2. Python 3.13 compatibility requires https://github.com/spulec/freezegun/pull/534 patch.
    # So >= 1.5.0 is needed. Currently (1.5.1), the #529 is not fixed, so I am pinning it to >=1.5.0, <= 1.5.1.
    try:
        current_time = asyncio.get_running_loop().time()
    except RuntimeError:  # Raised if there is no running loop
        loop = asyncio.new_event_loop()
        try:
            current_time = loop.run_until_complete(_time_coroutine())
        finally:
            loop.close()
    return current_time
