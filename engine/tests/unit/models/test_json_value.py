from __future__ import annotations

from engine.models.json_value import JsonMapping, JsonValue


def test_json_value_accepts_primitive_and_nested_shapes() -> None:
    value: JsonValue = {"a": [1, 2.0, True, None, "x", {"nested": [1]}]}
    assert isinstance(value, dict)


def test_json_mapping_alias_is_dict() -> None:
    mapping: JsonMapping = {"k": "v"}
    assert mapping["k"] == "v"
