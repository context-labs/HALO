import importlib
import inspect
import os
from collections import defaultdict
from collections.abc import Callable
from functools import cache
from typing import Any, cast

import yaml

from appworld.common.collections import subtract_lists
from appworld.common.imports import import_apis_module, import_models_module, import_tests_module
from appworld.common.io import read_file


def get_file_and_line_location(object: Any) -> str:
    """Get the file and line location of an object."""
    try:
        file_location = os.path.relpath(inspect.getfile(object))
        line_number = inspect.getsourcelines(object)[1]
        return f"{file_location}:{line_number}"
    except Exception:
        return ""


def get_line_number(class_or_function: type | Callable[..., Any]) -> int:
    return inspect.getsourcelines(class_or_function)[1]


def get_argument_names(function_or_class: Callable[..., Any] | type) -> list[str]:
    is_class = isinstance(function_or_class, type)
    if is_class:
        function_or_class = cast(Any, function_or_class).__init__
    arguments = list(inspect.signature(cast(Any, function_or_class)).parameters.keys())
    if is_class:
        arguments = subtract_lists(arguments, ["self", "cls"])
    return arguments


def get_name_to_model(app_name: str) -> dict[str, type]:
    from appworld.apps.lib.models.orm import SQLModel  # to avoid circular import.

    models_module = import_models_module(app_name)
    name_to_model = {
        model_name: model
        for model_name, model in inspect.getmembers(models_module, inspect.isclass)
        if issubclass(model, SQLModel) and model.__name__ not in ("SQLModel", "ModelHash")
    }
    return name_to_model


def get_model_name_to_line_number(app_name: str) -> dict[str, int]:
    name_to_model = get_name_to_model(app_name)
    return {name: get_line_number(model) for name, model in name_to_model.items()}


@cache
def get_name_to_apis(
    app_name: str, ignore_private_apis: bool = True
) -> dict[str, Callable[..., Any]]:
    apis_module = import_apis_module(app_name)

    def is_api_function(obj: type | Callable[..., Any]) -> bool:
        return inspect.isfunction(obj) and any(
            route.endpoint == obj for route in apis_module.app.routes
        )

    private_api_names = [
        "index",
        "set_local_home_path",
        "get_local_home_path",
        "set_date_and_time",
        "get_date_and_time",
    ]
    if app_name == "admin":
        private_api_names += [
            "show_card_balance",
            "validate_card",
            "make_deduction",
            "add_payment_card",
        ]

    name_to_apis = {
        name: obj
        for name, obj in inspect.getmembers(apis_module)
        if is_api_function(obj)
        if name not in private_api_names or not ignore_private_apis
    }
    return name_to_apis


def get_api_name_to_line_number(app_name: str) -> dict[str, int]:
    name_to_apis = get_name_to_apis(app_name)
    return {name: get_line_number(api) for name, api in name_to_apis.items()}


def get_api_name_to_test_names(app_name: str) -> dict[str, list[str]]:
    app_test_module = import_tests_module(app_name)
    test_classes = [
        test_class
        for name, test_class in inspect.getmembers(app_test_module, inspect.isclass)
        if name.startswith("Test") and name != "TestApp"
    ]
    if len(test_classes) != 1:
        raise Exception("Correct TestApp class couldn't be obtained.")

    test_app = test_classes[0]()
    test_methods = [
        getattr(test_app, method_name)
        for method_name in dir(test_app)
        if inspect.ismethod(getattr(test_app, method_name)) and method_name.startswith("test_")
    ]
    api_name_to_test_names: dict[str, list[str]] = {}
    test_module_code = inspect.getsource(app_test_module)
    test_methods = sorted(test_methods, key=lambda method: test_module_code.index(method.__name__))
    for method in test_methods:
        if not hasattr(method, "pytestmark"):
            raise Exception(f"The test {method.__name__} is not marked with @pytest.mark.api(...).")

        has_api_marker = False
        for marker in method.pytestmark:
            if marker.name == "api":
                api_name = marker.args[0]
                test_name = method.__name__
                if api_name in api_name_to_test_names:
                    api_name_to_test_names[api_name].append(test_name)
                else:
                    api_name_to_test_names[api_name] = [test_name]
                has_api_marker = True
        assert (
            has_api_marker
        ), f"The test {method.__name__} is not marked with @pytest.mark.api(...)."
    sorted_api_names = list(api_name_to_test_names.keys())
    app_apis_module = import_apis_module(app_name)
    apis_module_code = inspect.getsource(app_apis_module)
    sorted_api_names = sorted(
        sorted_api_names, key=lambda api_name: apis_module_code.index("def " + api_name)
    )
    api_name_to_test_names = {
        api_name: api_name_to_test_names[api_name] for api_name in sorted_api_names
    }
    return api_name_to_test_names


def get_model_name_to_api_names(app_name: str) -> dict[str, list[str]]:
    apis = import_apis_module(app_name)
    model_name_to_api_names: dict[str, list[str]] = defaultdict(list)
    for _, function in inspect.getmembers(apis):
        if not inspect.isfunction(function):
            continue
        if "@app." not in inspect.getsource(function):
            continue
        if not function.__doc__:
            continue
        try:
            spec_info = yaml.safe_load(function.__doc__.split("---")[0].rstrip())
            model_name = spec_info["model"]
            api_name = function.__name__
            model_name_to_api_names[model_name].append(api_name)
        except yaml.YAMLError:
            continue
    return dict(model_name_to_api_names)


def get_test_name_source_code(app_name: str, test_name: str) -> str:
    app_test_module = import_tests_module(app_name)
    test_classes = [
        test_class
        for name, test_class in inspect.getmembers(app_test_module, inspect.isclass)
        if name.startswith("Test") and name != "TestApp"
    ]
    if len(test_classes) != 1:
        raise Exception("Correct TestApp class couldn't be obtained.")

    test_app = test_classes[0]()
    test_method = getattr(test_app, test_name)
    return inspect.getsource(test_method)


def analyze_lru_cache_usage(directory_or_file_path: str) -> None:
    if directory_or_file_path.startswith("/"):
        raise Exception(
            "Absolute paths are not allowed. "
            "Pass them to be relative to the project root so that module imports can be done."
        )
    if not os.path.exists(directory_or_file_path):
        raise Exception(f"The directory or file path ({directory_or_file_path}) doesn't exist.")
    module_file_paths: list[str] = []
    if os.path.isfile(directory_or_file_path):
        if not directory_or_file_path.endswith(".py"):
            raise Exception(
                f"The file path ({directory_or_file_path}) is not a python file. "
                "Only python files are supported."
            )
        content = read_file(directory_or_file_path)
        if "@lru_cache" not in content:
            return
        module_file_paths.append(directory_or_file_path)
    else:
        for _, _, file_paths in os.walk(directory_or_file_path):
            for file_path in file_paths:
                if not file_path.endswith(".py"):
                    continue
                content = read_file(file_path)
                if "@lru_cache" not in content:
                    continue
                module_file_paths.append(file_path)
    for module_file_path in module_file_paths:
        module_path = module_file_path.replace(".py", "").replace("/", ".")
        module = importlib.import_module(module_path)
        print(f"Checking @lru_cache usage in: {module_file_path}")
        for name, obj in inspect.getmembers(module):
            if not hasattr(obj, "cache_info"):
                continue
            cache_info = obj.cache_info()
            if not hasattr(cache_info, "currsize"):
                continue
            print(f"Function: {name} (Cache size: {cache_info.currsize})")
