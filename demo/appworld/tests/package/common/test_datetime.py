from appworld.common.datetime import VanillaDateTime, VanillaTime


class TestCommonDatetime:
    def test_datetime_augmentation(self) -> None:
        import pendulum
        from pendulum import DateTime, Time

        assert pendulum.now().tz is None
        assert VanillaDateTime.now().tz is None
        assert DateTime.now().tz is None
        assert set(dir(DateTime)) == set(dir(DateTime))
        assert set(dir(Time)) == set(dir(Time))
        extra_datetime_attributes = {
            "add_range",
            "to_datetime",
            "sometime_this_week_in",
            "set_time",
            "sometime_next_month",
            "sometime_last_week",
            "subtract_range",
            "tomorrow",
            "from_month_string",
            "sometime_next_year",
            "add_max",
            "from_datetime_string",
            "is_time",
            "humanized_day_of_week",
            "from_datetime",
            "from_date_string",
            "sometime_this_month_in",
            "sometime_next_week_in",
            "sometime_last_month_in",
            "sometime_next_month_in",
            "sometime_this_month",
            "parse",
            "sometime_last_week_in",
            "sometime_this_year",
            "sometime_this_week",
            "last",
            "subtract_max",
            "yesterday",
            "sometime_last_year",
            "sometime_last_month",
            "datetime",
            "round",
            "to_month_string",
            "sometime_next_week",
        }
        assert set(dir(DateTime)) - set(dir(VanillaDateTime)) == extra_datetime_attributes
        extra_time_attributes = {"from_string", "to_string"}
        assert set(dir(Time)) - set(dir(VanillaTime)) == extra_time_attributes
        for attribute in extra_datetime_attributes:
            assert hasattr(DateTime.now(), attribute)
            assert hasattr(DateTime.today(), attribute)
            assert not hasattr(VanillaDateTime.now(), attribute)
            assert not hasattr(VanillaDateTime.today(), attribute)
        for attribute in extra_time_attributes:
            assert hasattr(DateTime.now().time(), attribute)
            assert not hasattr(VanillaDateTime.now().time(), attribute)
