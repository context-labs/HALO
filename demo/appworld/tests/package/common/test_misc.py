from appworld.common.misc import build_cli_command, rerank_by_similarity


class TestCommonMisc:
    def test_rerank_by_similarity(self) -> None:
        items = [
            {"name": "apple", "description": "A round fruit"},
            {"name": "banana", "description": "A long yellow fruit"},
            {"name": "grape", "description": "A small round fruit"},
        ]
        query = "round fruit"
        sorted_items = rerank_by_similarity(items, query, "description", "string")
        # Check if the items with "round" in their description come first
        assert sorted_items[0]["name"] == "apple" or sorted_items[0]["name"] == "grape"
        assert sorted_items[1]["name"] == "apple" or sorted_items[1]["name"] == "grape"
        assert sorted_items[2]["name"] == "banana"
        items = []
        query = "fruit"
        sorted_items = rerank_by_similarity(items, query, "description", "string")
        assert sorted_items == []

    def test_build_cli_command(self) -> None:
        # simple command with some arguments
        command = "my_command"
        arguments = {
            "bool_1": True,
            "bool_2": False,
            "int_1": 42,
            "float_1": 3.14,
            "str_1": "hello",
        }
        expected_output = 'my_command --bool-1 --int-1 42 --float-1 3.14 --str-1 "hello"'
        assert build_cli_command(command, arguments) == expected_output
        assert build_cli_command(command, arguments, use_hyphens=True) == expected_output
        # command with no arguments
        command = "my_command"
        arguments = {}
        expected_output = "my_command"
        assert build_cli_command(command, arguments) == expected_output
        # only arguments with no command
        command = ""
        arguments = {
            "bool_1": True,
            "int_1": 42,
            "float_1": 3.14,
            "str_1": "hello",
        }
        expected_output = '--bool-1 --int-1 42 --float-1 3.14 --str-1 "hello"'
        assert build_cli_command(command, arguments) == expected_output
        # command with use_hyphens=False
        command = "my_command"
        arguments = {
            "bool_1": True,
            "bool_2": False,
            "int_1": 42,
            "float_1": 3.14,
            "str_1": "hello",
        }
        expected_output = 'my_command --bool_1 --int_1 42 --float_1 3.14 --str_1 "hello"'
        assert build_cli_command(command, arguments, use_hyphens=False) == expected_output
