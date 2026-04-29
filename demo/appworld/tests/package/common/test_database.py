from appworld.common.database import search_fts


class TestCommonDatabase:
    def test_search_fts(self) -> None:
        items = ["apple", "banana", "cherries"]
        assert search_fts(query="apples", items=items) == ["apple", "banana", "cherries"]
        assert search_fts(query="banana", items=items) == ["banana", "apple", "cherries"]
        assert search_fts(query="cherry", items=items) == ["cherries", "apple", "banana"]
