import inspect
import random
import re
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime as _datetime
from typing import Any, ClassVar, Self, cast

import pendulum
from pendulum import Date as _Date
from pendulum import Time as _Time
from pendulum.datetime import DateTime as _DateTime
from pydantic._internal._generate_schema import GenerateSchema
from pydantic_core import core_schema

from appworld.common.copy import deep_copy_class


WEEK_DAY = int  # 0-6 (from PENDULUM_DAYS)
MONTH_DAY = int  # 1-31 (max is month dependent)

PENDULUM_DAYS = [
    pendulum.SUNDAY,  # 0
    pendulum.MONDAY,  # 1
    pendulum.TUESDAY,  # 2
    pendulum.WEDNESDAY,  # 3
    pendulum.THURSDAY,  # 4
    pendulum.FRIDAY,  # 5
    pendulum.SATURDAY,  # 6
]

initial_match_type = cast(Any, GenerateSchema.match_type)


def match_type(self: GenerateSchema, obj: Any) -> Any:
    # Patch as per: https://github.com/pydantic/pydantic/discussions/9343
    if getattr(obj, "__name__", "").lower() == "datetime":
        return core_schema.datetime_schema()
    return initial_match_type(self, obj)


setattr(GenerateSchema, "match_type", cast(Any, match_type))  # noqa


class WeekDay:
    days: ClassVar[dict[int, str]] = {
        pendulum.MONDAY: "Monday",
        pendulum.TUESDAY: "Tuesday",
        pendulum.WEDNESDAY: "Wednesday",
        pendulum.THURSDAY: "Thursday",
        pendulum.FRIDAY: "Friday",
        pendulum.SATURDAY: "Saturday",
        pendulum.SUNDAY: "Sunday",
    }
    name_formats: ClassVar[dict[str, str]] = {
        "mon": "Monday",
        "monday": "Monday",
        "tue": "Tuesday",
        "tuesday": "Tuesday",
        "wed": "Wednesday",
        "wednesday": "Wednesday",
        "thu": "Thursday",
        "thursday": "Thursday",
        "fri": "Friday",
        "friday": "Friday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }

    def __init__(self, value: int | str):
        if isinstance(value, int):
            if value < 0 or value > 6:
                raise ValueError("Invalid day number. Day number should be between 0 and 6.")
            self.value = value
        elif isinstance(value, str):
            normalized_value = value.strip().lower()
            if normalized_value not in WeekDay.days.values():
                if normalized_value in WeekDay.name_formats:
                    normalized_value = WeekDay.name_formats[normalized_value]
                else:
                    raise ValueError(f"Invalid day name: {value}. Please use a valid day name.")
            self.value = self._get_day_number(normalized_value)
        else:
            raise ValueError("Invalid input. Please provide a valid day number or name.")

    def number(self) -> int:
        return int(self.value)

    def word(self, shorten: bool = False) -> str:
        word_ = WeekDay.days[self.value]
        if shorten:
            word_ = word_[:3]
        return word_

    def _get_day_number(self, day_name: str) -> int:
        for num, name in WeekDay.days.items():
            if name == day_name:
                return num
        raise ValueError(f"Invalid day name: {day_name}")


class Time(_Time):
    @classmethod
    def from_string(cls, time_string: str) -> Self:
        from appworld.apps.lib.apis.datetime import parse_time

        time = parse_time(time_string)
        return cls(time.hour, time.minute, time.second)

    def to_string(self) -> str:
        from appworld.apps.lib.apis.datetime import time_to_string

        return time_to_string(cast(Any, self))


pendulum_parse = pendulum.parse
pendulum_instance = pendulum.instance


class PendulumDateTime(_DateTime):
    """
    Barebone override of pendulum.DateTime to remove timezone.
    Models can use this class because it works exactly the
    same as pendulum.DateTime, except it doesn't have timezone.

    Loading this file also monkey patches pendulum.DateTime
    to this class.
    """

    @classmethod
    def now(cls, tz: Any | None = None) -> Self:
        return cast(Self, pendulum_instance(_datetime.now(), tz=None))

    @classmethod
    def strptime(cls, time: str, fmt: str) -> Self:
        return cast(Self, DateTime.from_datetime(_datetime.strptime(time, fmt)))

    @classmethod
    def today(cls) -> Self:
        return cls.now()

    @classmethod
    def instance(cls, dt: Any, tz: Any | None = None) -> Self:
        _ = tz  # do not use tz
        return super().instance(dt, tz=None)


class DateTime(PendulumDateTime):
    @classmethod
    def datetime(
        cls,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> Self:
        return cast(
            Self,
            pendulum_instance(
                _datetime(year, month, day, hour, minute, second, microsecond),
                tz=None,
            ),
        )

    @classmethod
    def parse(cls, *args: Any, **kwargs: Any) -> Self:
        kwargs.pop("tz", None)
        result = pendulum_parse(*args, **kwargs, tz=None)
        assert isinstance(result, _DateTime)
        return cast(Self, result)

    @classmethod
    def from_datetime(cls, datetime: _datetime) -> Self:
        return cast(Self, pendulum_instance(datetime, tz=None))

    def to_datetime(self) -> _datetime:
        return _datetime(
            year=self.year,
            month=self.month,
            day=self.day,
            hour=self.hour,
            minute=self.minute,
            second=self.second,
            microsecond=self.microsecond,
        )

    def add_max(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
    ) -> Self:
        seconds += minutes * 60
        seconds += hours * 60 * 60
        seconds += days * 24 * 60 * 60
        seconds += weeks * 7 * 24 * 60 * 60
        seconds += months * 30 * 24 * 60 * 60
        seconds += years * 365 * 24 * 60 * 60
        seconds = random.randint(0, seconds)
        return self.add(seconds=seconds)

    def subtract_max(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
    ) -> Self:
        seconds += minutes * 60
        seconds += hours * 60 * 60
        seconds += days * 24 * 60 * 60
        seconds += weeks * 7 * 24 * 60 * 60
        seconds += months * 30 * 24 * 60 * 60
        seconds += years * 365 * 24 * 60 * 60
        seconds = random.randint(0, seconds)
        return self.subtract(seconds=seconds)

    def subtract_range(
        self,
        years: tuple[int, int] = (0, 0),
        months: tuple[int, int] = (0, 0),
        weeks: tuple[int, int] = (0, 0),
        days: tuple[int, int] = (0, 0),
        hours: tuple[int, int] = (0, 0),
        minutes: tuple[int, int] = (0, 0),
        seconds: tuple[int, int] = (0, 0),
    ) -> Self:
        for tuple_ in (years, months, weeks, days, hours, minutes, seconds):
            if len(tuple_) != 2:
                raise ValueError(
                    f"Invalid tuple length in subtract_range. Expected 2, found {len(tuple_)}"
                )
            if tuple_[0] > tuple_[1]:
                raise ValueError(
                    "Invalid tuple order in subtract_range. Expected tuple_[0] <= tuple_[1], "
                    f"found {tuple_[0]} > {tuple_[1]}"
                )
        datetime_: Self = self.subtract(
            years=years[0],
            months=months[0],
            weeks=weeks[0],
            days=days[0],
            hours=hours[0],
            minutes=minutes[0],
            seconds=seconds[0],
        )
        datetime_ = datetime_.subtract_max(
            years=years[1] - years[0],
            months=months[1] - months[0],
            weeks=weeks[1] - weeks[0],
            days=days[1] - days[0],
            hours=hours[1] - hours[0],
            minutes=minutes[1] - minutes[0],
            seconds=seconds[1] - seconds[0],
        )
        return datetime_

    def add_range(
        self,
        years: tuple[int, int] = (0, 0),
        months: tuple[int, int] = (0, 0),
        weeks: tuple[int, int] = (0, 0),
        days: tuple[int, int] = (0, 0),
        hours: tuple[int, int] = (0, 0),
        minutes: tuple[int, int] = (0, 0),
        seconds: tuple[int, int] = (0, 0),
    ) -> Self:
        for tuple_ in (years, months, weeks, days, hours, minutes, seconds):
            if len(tuple_) != 2:
                raise ValueError(
                    f"Invalid tuple length in add_range. Expected 2, found {len(tuple_)}"
                )
            if tuple_[0] > tuple_[1]:
                raise ValueError(
                    "Invalid tuple order in add_range. Expected tuple_[0] <= tuple_[1], "
                    f"found {tuple_[0]} > {tuple_[1]}"
                )
        datetime_: Self = self.add(
            years=years[0],
            months=months[0],
            weeks=weeks[0],
            days=days[0],
            hours=hours[0],
            minutes=minutes[0],
            seconds=seconds[0],
        )
        datetime_ = datetime_.add_max(
            years=years[1] - years[0],
            months=months[1] - months[0],
            weeks=weeks[1] - weeks[0],
            days=days[1] - days[0],
            hours=hours[1] - hours[0],
            minutes=minutes[1] - minutes[0],
            seconds=seconds[1] - seconds[0],
        )
        return datetime_

    def set_time(self, time: str, before: _Time | None = None, after: _Time | None = None) -> Self:
        from appworld.common.random import choose_from_time_range  # to avoid circular import

        date = self
        if re.match(r"^\d{2}:\d{2}$", time):
            time = f"{time}:00"

        if re.match(r"^\d{2}:\d{2}:\d{2}$", time):
            if before is not None or after is not None:
                raise Exception(
                    f"The time {time} is specified in HH:MM:SS format. Yet before/after are passed."
                )
            components = time.split(":")
            for component in components:
                assert component.isdigit(), f"Invalid time: {time}"
            hour, minute, second = (int(component) for component in components)
            if hour > 23:
                raise Exception(f"Invalid hour: {hour} in time: {time}")
            if minute > 59:
                raise Exception(f"Invalid minute: {minute} in time: {time}")
            if second > 59:
                raise Exception(f"Invalid second: {second} in time: {time}")
            date = date.set(hour=hour, minute=minute, second=second, microsecond=0)
            return date
        elif time not in ("same", "any", "day", "morning", "afternoon", "evening", "night"):
            raise Exception(
                f"Invalid time: {time}. It should be either in HH:MM:SS format or HH:MM format or one of "
                "'same', 'any', 'day', 'morning', 'afternoon', 'evening', 'night'."
            )
        if before is not None and after is not None and before <= after:
            raise Exception(
                f"Invalid time range: before ({before}) should be more than after ({after})."
            )
        if isinstance(before, _DateTime):
            raise Exception(
                "`before` should be a pendulum Time object, not a DateTime object. "
                "You can obtain it by calling .time() on a DateTime object."
            )
        if isinstance(after, _DateTime):
            raise Exception(
                "`after` should be a pendulum Time object, not a DateTime object. "
                "You can obtain it by calling .time() on a DateTime object."
            )
        if time == "same":
            pass
        elif time == "any":
            # 12am - 11:59pm
            min_time, max_time = _Time(0, 0, 0), _Time(23, 59, 59)
            min_time = after if after is not None else min_time
            max_time = before if before is not None else max_time
            time_ = choose_from_time_range(min_time, max_time)
            date = date.set(hour=time_.hour)
            date = date.set(minute=time_.minute)
            date = date.set(second=time_.second)
        elif time == "day":
            # 8am - 7pm
            min_time, max_time = _Time(8, 0, 0), _Time(19, 0, 0)
            min_time = after if after is not None else min_time
            max_time = before if before is not None else max_time
            time_ = choose_from_time_range(min_time, max_time)
            date = date.set(hour=time_.hour)
            date = date.set(minute=time_.minute)
            date = date.set(second=time_.second)
            assert date.is_time("day")  # TODO: move to tests/
        elif time == "morning":
            # 8am - 12pm
            min_time, max_time = _Time(8, 0, 0), _Time(12, 0, 0)
            min_time = after if after is not None else min_time
            max_time = before if before is not None else max_time
            time_ = choose_from_time_range(min_time, max_time)
            date = date.set(hour=time_.hour)
            date = date.set(minute=time_.minute)
            date = date.set(second=time_.second)
            assert date.is_time("morning")  # TODO: move to tests/
        elif time == "afternoon":
            # 12pm - 5pm
            min_time, max_time = _Time(12, 0, 0), _Time(17, 0, 0)
            min_time = after if after is not None else min_time
            max_time = before if before is not None else max_time
            time_ = choose_from_time_range(min_time, max_time)
            date = date.set(hour=time_.hour)
            date = date.set(minute=time_.minute)
            date = date.set(second=time_.second)
            assert date.is_time("afternoon")  # TODO: move to tests/
        elif time == "evening":
            # 5pm - 8pm
            min_time, max_time = _Time(17, 0, 0), _Time(20, 0, 0)
            min_time = after if after is not None else min_time
            max_time = before if before is not None else max_time
            time_ = choose_from_time_range(min_time, max_time)
            date = date.set(hour=time_.hour)
            date = date.set(minute=time_.minute)
            date = date.set(second=time_.second)
            assert date.is_time("evening")  # TODO: move to tests/
        elif time == "night":
            # 8pm - 12am
            min_time, max_time = _Time(20, 0, 0), _Time(23, 0, 0)
            min_time = after if after is not None else min_time
            max_time = before if before is not None else max_time
            time_ = choose_from_time_range(min_time, max_time)
            date = date.set(hour=time_.hour)
            date = date.set(minute=time_.minute)
            date = date.set(second=time_.second)
            assert date.is_time("night")  # TODO: move to tests/
        else:
            raise Exception(f"Invalid time: {time}")
        return date

    def is_time(self, time: str) -> bool:
        assert time in ("day", "morning", "afternoon", "evening", "night")
        if time == "day":
            condition_1 = 8 <= self.hour <= 19
            condition_2 = self.hour == 20 and self.minute == 0 and self.second == 0
        elif time == "morning":
            condition_1 = 8 <= self.hour <= 11
            condition_2 = self.hour == 12 and self.minute == 0 and self.second == 0
        elif time == "afternoon":
            condition_1 = 12 <= self.hour <= 16
            condition_2 = self.hour == 17 and self.minute == 0 and self.second == 0
        elif time == "evening":
            condition_1 = 17 <= self.hour <= 19
            condition_2 = self.hour == 20 and self.minute == 0 and self.second == 0
        elif time == "night":
            condition_1 = 20 <= self.hour <= 23
            condition_2 = self.hour == 0 and self.minute == 0 and self.second == 0
        else:
            raise Exception(f"Invalid time: {time}")
        return condition_1 or condition_2

    @classmethod
    def today(cls, time: str = "same") -> Self:
        date0 = cls.now()
        date1 = date0.set_time(time)
        return date1

    @classmethod
    def tomorrow(cls, time: str = "same") -> Self:
        date0 = cls.now().add(days=1)
        date1 = date0.set_time(time)
        return date1

    @classmethod
    def yesterday(cls, time: str = "same") -> Self:
        date0 = cls.now().subtract(days=1)
        date1 = date0.set_time(time)
        return date1

    def last(self, *args: Any, **kwargs: Any) -> Self:
        # alias for previous (as other methods use last instead of previous)
        return self.previous(*args, **kwargs)

    def sometime_this_year(self, direction: str | None = None, time: str = "same") -> Self:
        assert direction in ("past", "future", None)
        first_day_of_year = self.start_of("year")
        if self.day_of_year == 1 and direction == "past":
            raise Exception(
                "Today is the first day of the year, so can't get a random date in the past of this year."
            )
        if self.day_of_year == 365 and direction == "future":
            raise Exception(
                "Today is the last day of the year, so can't get a random date in the future of this year."
            )
        if direction == "past" and self.day_of_year > 1:
            datetime_ = first_day_of_year.add(days=random.randint(1, self.day_of_year - 1))
        elif direction == "future" and self.day_of_year < 365:
            days_remaining = 365 - self.day_of_year
            datetime_ = first_day_of_year.add(days=days_remaining).add(
                days=random.randint(1, days_remaining)
            )
        else:
            datetime_ = first_day_of_year.add(days=random.randint(0, 364))
        return datetime_.set_time(time)

    def sometime_last_year(self, time: str = "same") -> Self:
        first_day_of_last_year = self.start_of("year").subtract(years=1)
        datetime_ = first_day_of_last_year.add(days=random.randint(0, 364))
        return datetime_.set_time(time)

    def sometime_next_year(self, time: str = "same") -> Self:
        first_day_of_next_year = self.start_of("year").add(years=1)
        datetime_ = first_day_of_next_year.add(days=random.randint(0, 364))
        return datetime_.set_time(time)

    def sometime_this_month(self, direction: str | None = None, time: str = "same") -> Self:
        assert direction in ("past", "future", None)
        first_day_of_month = self.start_of("month")
        last_day_of_month = self.end_of("month")
        if self.day == first_day_of_month.day and direction == "past":
            raise Exception(
                "Today is the first day of the month, so can't get a random date in the past of this month."
            )
        if self.day == last_day_of_month.day and direction == "future":
            raise Exception(
                "Today is the last day of the month, so can't get a random date in the future of this month."
            )
        if direction == "past" and self.day > 1:
            datetime_ = first_day_of_month.add(days=random.randint(1, self.day - 1))
        elif direction == "future" and self.day < self.days_in_month:
            days_remaining = self.days_in_month - self.day
            datetime_ = first_day_of_month.add(days=days_remaining).add(
                days=random.randint(1, days_remaining)
            )
        else:
            datetime_ = first_day_of_month.add(days=random.randint(0, self.days_in_month - 1))
        return datetime_.set_time(time)

    def sometime_last_month(self, time: str = "same") -> Self:
        first_day_of_last_month = self.subtract(months=1).start_of("month")
        datetime_ = first_day_of_last_month.add(days=random.randint(0, self.days_in_month - 1))
        return datetime_.set_time(time)

    def sometime_next_month(self, time: str = "same") -> Self:
        first_day_of_next_month = self.add(months=1).start_of("month")
        datetime_ = first_day_of_next_month.add(days=random.randint(0, self.days_in_month - 1))
        return datetime_.set_time(time)

    def sometime_this_week(self, direction: str | None = None, time: str = "same") -> Self:
        assert direction in ("past", "future", None)
        pendulum_days = deepcopy(PENDULUM_DAYS)
        today_day = self.day_of_week
        if today_day == 0 and direction == "past":
            raise Exception("Today is Sunday, so can't get a random date in the past of this week.")
        if today_day == 6 and direction == "future":
            raise Exception(
                "Today is Saturday, so can't get a random date in the future of this week."
            )
        if direction == "future":
            pendulum_days = pendulum_days[today_day:]
        elif direction == "past":
            pendulum_days = pendulum_days[: today_day + 1]
        chosen_day = random.choice(pendulum_days)
        date0 = self.next(cast(Any, chosen_day))
        date1 = date0.set_time(time)
        return date1

    def sometime_last_week(self, time: str = "same") -> Self:
        last_week_start = self.subtract(weeks=1).start_of("week")
        datetime_ = last_week_start.add(days=random.randint(0, 6))
        return datetime_.set_time(time)

    def sometime_next_week(self, time: str = "same") -> Self:
        next_week_start = self.add(weeks=1).start_of("week")
        datetime_ = next_week_start.add(days=random.randint(0, 6))
        return datetime_.set_time(time)

    def sometime_this_week_in(self, days: list[WEEK_DAY], time: str = "same") -> Self:
        for day in days:
            assert day in range(0, 7), f"Invalid day: {day}"
        date0 = self.start_of("week").next(cast(Any, random.choice(days)))
        date1 = date0.set_time(time)
        return date1

    def sometime_last_week_in(self, days: list[WEEK_DAY], time: str = "same") -> Self:
        for day in days:
            assert day in range(0, 7), f"Invalid day: {day}"
        date0 = self.start_of("week").previous(cast(Any, random.choice(days)))
        date1 = date0.set_time(time)
        return date1

    def sometime_next_week_in(self, days: list[WEEK_DAY], time: str = "same") -> Self:
        for day in days:
            assert day in range(0, 7), f"Invalid day: {day}"
        date0 = self.end_of("week").next(cast(Any, random.choice(days)))
        date1 = date0.set_time(time)
        return date1

    def sometime_last_month_in(self, days: list[MONTH_DAY], time: str = "same") -> Self:
        last_month_date = self.start_of("month").subtract(days=1)
        for day in days:
            assert day in range(1, last_month_date.days_in_month + 1), f"Invalid day: {day}"
        datetime_ = last_month_date.set(day=random.choice(days))
        return datetime_.set_time(time)

    def sometime_this_month_in(self, days: list[MONTH_DAY], time: str = "same") -> Self:
        for day in days:
            assert day in range(1, self.days_in_month + 1), f"Invalid day: {day}"
        datetime_ = self.set(day=random.choice(days))
        return datetime_.set_time(time)

    def sometime_next_month_in(self, days: list[MONTH_DAY], time: str = "same") -> Self:
        next_month_date = self.end_of("month").add(days=1)
        for day in days:
            assert day in range(1, next_month_date.days_in_month + 1), f"Invalid day: {day}"
        datetime_ = next_month_date.set(day=random.choice(days))
        return datetime_.set_time(time)

    @classmethod
    def from_month_string(cls, month_string: str, day: int = 1, time: str = "day") -> Self:
        assert time in ("day", "morning", "afternoon", "evening", "night")
        date_ = cls.strptime(month_string, "%m/%Y")
        date_ = date_.set(day=day)
        date_ = date_.set_time(time)
        return date_

    def to_month_string(self) -> str:
        return self.strftime("%Y-%m").lower()

    @classmethod
    def from_date_string(cls, date_string: str, time: str = "day") -> Self:
        from appworld.apps.lib.apis.datetime import parse_date  # to avoid circular import

        assert time in ("day", "morning", "afternoon", "evening", "night")
        date0 = cls.from_datetime(parse_date(date_string))
        date1 = date0.set_time(time)
        return date1

    def to_date_string(self) -> str:
        from appworld.apps.lib.apis.datetime import date_to_string  # to avoid circular import

        return date_to_string(self)

    @classmethod
    def from_datetime_string(cls, datetime_string: str) -> Self:
        from appworld.apps.lib.apis.datetime import parse_datetime  # to avoid circular import

        datetime = cls.from_datetime(parse_datetime(datetime_string))
        return datetime

    def to_datetime_string(self) -> str:
        from appworld.apps.lib.apis.datetime import datetime_to_string  # to avoid circular import

        return datetime_to_string(self)

    def to_time_string(self) -> str:
        from appworld.apps.lib.apis.datetime import time_to_string  # to avoid circular import

        return time_to_string(cast(Any, self))

    def humanized_day_of_week(self, shorten: bool = False) -> str:
        word = self.format("dddd")
        if shorten:
            word = word[:3]
        return word

    def round(self, to: str) -> Self:
        assert to in ("second", "minute", "hour", "day")
        if to == "second":
            return self.add(microseconds=500_000).start_of("second")
        elif to == "minute":
            return self.add(seconds=30).start_of("minute")
        elif to == "hour":
            return self.add(minutes=30).start_of("hour")
        elif to == "day":
            return self.add(hours=12).start_of("day")
        raise ValueError(f"Invalid round unit: {to}")  # mypy

    @classmethod
    def fromisoformat(cls, *args: Any, **kwargs: Any) -> Self:
        kwargs.pop("tz", None)
        result = super().fromisoformat(*args, **kwargs)
        return result

    def add(self, *args: Any, **kwargs: Any) -> Self:
        return super().add(*args, **kwargs)

    def subtract(self, *args: Any, **kwargs: Any) -> Self:
        return super().subtract(*args, **kwargs)


VanillaDateTime = deep_copy_class(cast(type[type], PendulumDateTime), name="DateTime")
VanillaDate = deep_copy_class(cast(type[type], pendulum.Date), name="Date")
VanillaTime = deep_copy_class(cast(type[type], pendulum.Time), name="Time")


def vanilla_converter(function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        object_ = function(*args, **kwargs)
        if isinstance(object_, _DateTime):
            return cast(Any, VanillaDateTime)(
                object_.year,
                object_.month,
                object_.day,
                object_.hour,
                object_.minute,
                object_.second,
                object_.microsecond,
            )
        if isinstance(object_, _Date):
            return cast(Any, VanillaDate)(object_.year, object_.month, object_.day)
        if isinstance(object_, _Time):
            return cast(Any, VanillaTime)(object_.hour, object_.minute, object_.second)
        return object_

    return wrapper


for klass in (VanillaDateTime, VanillaDate, VanillaTime):
    for attr_name in dir(klass):
        attr_value = getattr(klass, attr_name)
        if inspect.ismethod(attr_value) or inspect.isfunction(attr_value):
            setattr(klass, attr_name, vanilla_converter(attr_value))

pendulum.DateTime = PendulumDateTime  # type: ignore[misc]

attributes = [
    "add_max",
    "subtract_max",
    "add_range",
    "subtract_range",
    "set_time",
    "to_datetime",
    "now",
    "strptime",
    "today",
    "tomorrow",
    "yesterday",
    "sometime_this_year",
    "sometime_last_year",
    "sometime_next_year",
    "sometime_this_month",
    "sometime_last_month",
    "sometime_next_month",
    "sometime_this_week",
    "sometime_last_week",
    "sometime_next_week",
    "sometime_this_week_in",
    "sometime_last_week_in",
    "sometime_next_week_in",
    "sometime_last_month_in",
    "sometime_this_month_in",
    "sometime_next_month_in",
    "last",
    "to_month_string",
    "to_date_string",
    "to_datetime_string",
    "to_time_string",
    "is_time",
    "humanized_day_of_week",
    "round",
    "parse",
    "from_datetime",
    "from_month_string",
    "from_datetime_string",
    "from_date_string",
    "datetime",
]
for attribute in attributes:
    setattr(pendulum.DateTime, attribute, getattr(DateTime, attribute))
attributes = ["from_string", "to_string"]
for attribute in attributes:
    setattr(pendulum.Time, attribute, getattr(Time, attribute))
attributes = ["parse", "instance", "now", "today", "tomorrow", "yesterday"]
for attribute in attributes:
    setattr(pendulum, attribute, getattr(DateTime, attribute))


class SqlalchemyDateTimeProcessor:
    """
    To make sure the datetime objects returned from DB are of pendulum datetime type (not datetime.datetime).
    I have to make this custom processor, as I need them to not have timezone.
    https://github.com/yk-lab/sqlalchemy-utils/blob/2970e2d0380852ab554ae68d08fba7fa17b62b9d/sqlalchemy_utils/types/enriched_datetime/pendulum_datetime.py
    https://github.com/kvesteri/sqlalchemy-utils/blob/master/sqlalchemy_utils/types/enriched_datetime/enriched_datetime_type.py
    https://github.com/tiangolo/sqlmodel/issues/447
    """

    def _coerce(self, impl: Any, value: Any) -> Any:
        if value is not None:
            if isinstance(value, str) and is_iso_format(value):
                value = _datetime.fromisoformat(value)
            if isinstance(value, pendulum.DateTime):
                pass
            elif isinstance(value, _datetime):
                value = DateTime.from_datetime(value)
            else:
                raise Exception(f"Cannot parse to DateTime: {value}")
        return value

    def process_bind_param(self, impl: Any, value: Any, dialect: Any) -> Any:
        if value:
            return self._coerce(impl, value)
        return value

    def process_result_value(self, impl: Any, value: Any, dialect: Any) -> Any:
        if value:
            return DateTime.parse(value.isoformat())
        return value


def is_iso_format(string: str) -> bool:
    try:
        _datetime.fromisoformat(string)
        return True
    except ValueError:
        return False
