from appworld.common.datetime import DateTime
from appworld.common.io import JsonPlus


class TestCommonIO:
    def test_json_plus(self) -> None:
        data: list | dict = {
            1: "1",
            "2": 2,
            3.1: "3.1",
            4: DateTime.now(),
            DateTime.now(): DateTime.now(),
            DateTime.now().date(): [DateTime.now().date()],
            "nested": {
                1: "1",
                "2": 2,
                3.1: "3.1",
                4: [DateTime.now()],
                DateTime.now(): DateTime.now(),
                DateTime.now().date(): DateTime.now().date(),
            },
        }
        assert JsonPlus.loads(JsonPlus.dumps(data)) == data
        data = [data, data]
        assert JsonPlus.loads(JsonPlus.dumps(data)) == data
        data = [1, 2.3, DateTime.now(), DateTime.now()]
        assert JsonPlus.loads(JsonPlus.dumps(data)) == data
        data = (DateTime.now(),)
        assert JsonPlus.loads(JsonPlus.dumps(data)) == JsonPlus.copy(data)
        datetime_now = DateTime.now()
        data_in = {"123": ([datetime_now, datetime_now.date()],)}
        data_out = {"123": [[datetime_now, datetime_now.date()]]}
        assert JsonPlus.loads(JsonPlus.dumps(data_in)) == JsonPlus.copy(data_out)
