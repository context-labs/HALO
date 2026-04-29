import argparse
import os
from collections.abc import Sequence
from typing import Any
from typing import get_args as get_type_args

from appworld.common.collections import flatten, unique
from appworld.common.imports import ensure_package_installed
from appworld.common.io import read_file, write_file
from appworld.common.path_store import path_store
from appworld_agents.configs._generator.agents import (
    AGENT_INFOS,
    AGENT_NAME_TO_INFO,
    AgentInfo,
)
from appworld_agents.configs._generator.agents import PART_OF_LITERAL as AGENT_PART_OF_LITERAL
from appworld_agents.configs._generator.models import (
    MODEL_INFOS,
    MODEL_NAME_TO_INFO,
    ModelInfo,
)
from appworld_agents.configs._generator.models import PART_OF_LITERAL as MODEL_PART_OF_LITERAL


ensure_package_installed("_jsonnet")
ensure_package_installed("jinja2")
from jinja2 import BaseLoader, Environment, StrictUndefined  # noqa: E402


def read_template(name: str) -> str:
    file_path = os.path.join(
        path_store.experiment_configs, "_generator", "templates", name + ".jsonnet.j2"
    )
    output = read_file(file_path)
    assert isinstance(output, str)  # mypy
    return output


def render_template(template: str, **kwargs: Any) -> str:
    env = Environment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.from_string(template).render(**kwargs)


def strip_list(items: list[str]) -> list[str]:
    return [item.strip() for item in items]


def intersects(list1: Sequence[str], list2: Sequence[str]) -> bool:
    return bool(set(list1) & set(list2))


def build_config(agent_info: AgentInfo, model_info: ModelInfo, dataset_name: str) -> str:
    template = read_template(agent_info.agent_name)
    model_server_config = model_info.model_server_config
    if agent_info.function_calling and not model_info.function_calling:
        raise ValueError(
            f"Agent '{agent_info.agent_name}' requires a model with function calling capability, "
            f"but model '{model_info.model_name}' does not support it. Please choose a different model or agent."
        )
    rendered_config = render_template(
        template,
        agent_info=agent_info.model_dump(),
        model_info=model_info.model_dump(),
        model_server_config=model_server_config,
        dataset_name=dataset_name,
    )
    for literal, real in [
        ('"LITERAL_DOUBLE_QUOTE', '"'),
        ('LITERAL_DOUBLE_QUOTE"', '"'),
        ("'LITERAL_SINGLE_QUOTE", "'"),
        ("LITERAL_SINGLE_QUOTE'", "'"),
        ("LITERAL_DOUBLE_QUOTE", '"'),
        ("LITERAL_SINGLE_QUOTE", "'"),
    ]:
        rendered_config = rendered_config.replace(literal, real)
    return rendered_config


def make_experiment_name(
    creator_name: str,
    model_name: str,
    agent_name: str,
    dataset_name: str,
    flat: bool = False,
) -> str:
    experiment_name = os.path.sep.join([agent_name, creator_name, model_name, dataset_name])
    if flat:
        experiment_name = experiment_name.replace(os.path.sep, "_")
    return experiment_name


def make_experiment_config(
    model_name: str,
    agent_name: str,
    dataset_name: str,
    save: bool = True,
    dry_run: bool = False,
    flat: bool = False,
) -> dict[str, Any]:
    if model_name not in MODEL_NAME_TO_INFO:
        raise ValueError(f"Model {model_name} not found")
    selected_model_info: ModelInfo = MODEL_NAME_TO_INFO[model_name]
    if agent_name not in AGENT_NAME_TO_INFO:
        raise ValueError(f"Agent {agent_name} not found")
    selected_agent_info: AgentInfo = AGENT_NAME_TO_INFO[agent_name]
    rendered_config = build_config(
        agent_info=selected_agent_info,
        model_info=selected_model_info,
        dataset_name=dataset_name,
    )
    experiment_name = make_experiment_name(
        creator_name=selected_model_info.creator,
        model_name=model_name,
        agent_name=agent_name,
        dataset_name=dataset_name,
        flat=flat,
    )
    output_file_path = path_store.experiment_config_file_path(experiment_name)
    if dry_run:
        print(f"Would write: {output_file_path}")
    elif save:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        write_file(rendered_config, output_file_path)
        print(f"Written to: {output_file_path}")
    return {"name": experiment_name, "config": rendered_config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AppWorld experiment configs.")
    parser.add_argument(
        "--model_collection_names",
        type=str,
        default=None,
        help=(
            "Comma-separated list of model collection names from: "
            + ", ".join(get_type_args(MODEL_PART_OF_LITERAL))
        ),
    )
    parser.add_argument(
        "--model_names",
        type=str,
        default=None,
        help=(
            "Comma-separated list of model names. If passed, will limit to only these models from the collection."
        ),
    )
    parser.add_argument(
        "--agent_collection_names",
        type=str,
        default=None,
        help=(
            "Comma-separated list of agent collection names from: "
            + ", ".join(get_type_args(AGENT_PART_OF_LITERAL))
        ),
    )
    parser.add_argument(
        "--agent_names",
        type=str,
        default=None,
        help=(
            "Comma-separated list of agent short names. If passed, will limit to only these agents."
        ),
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        default="dev,test_normal,test_challenge",
        help="Comma-separated list of dataset names",
    )
    parser.add_argument("--prefix", type=str, help="Optional file name prefix", default="")
    parser.add_argument("--suffix", type=str, help="Optional file name suffix", default="")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If passed, will only print the files that would be written, without writing them.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="If passed, will write all files to the root configs directory instead of subdirectories.",
    )
    args = parser.parse_args()
    dataset_names = strip_list(args.dataset_names.split(","))
    model_collection_names = strip_list(
        args.model_collection_names.split(",")
        if args.model_collection_names
        else unique(flatten([e.part_of for e in MODEL_INFOS]))
    )
    agent_collection_names = strip_list(
        args.agent_collection_names.split(",")
        if args.agent_collection_names
        else unique(flatten([e.part_of for e in AGENT_INFOS]))
    )
    model_names = strip_list(
        args.model_names.split(",") if args.model_names else list(MODEL_NAME_TO_INFO.keys())
    )
    agent_names = strip_list(
        args.agent_names.split(",") if args.agent_names else list(AGENT_NAME_TO_INFO.keys())
    )
    for dataset_name in dataset_names:
        for model_info in MODEL_INFOS:
            model_part_of = model_info.part_of
            if model_collection_names and not intersects(model_part_of, model_collection_names):
                continue
            model_name = model_info.model_name
            if model_name not in model_names:
                continue
            for agent_info in AGENT_INFOS:
                agent_part_of = agent_info.part_of
                agent_name = agent_info.agent_name
                agent_function_calling = agent_info.function_calling
                model_function_calling = model_info.function_calling
                if not intersects(agent_part_of, agent_collection_names):
                    continue
                if agent_info.agent_name not in agent_names:
                    continue
                if not model_function_calling and agent_function_calling:
                    continue
                make_experiment_config(
                    model_name=model_name,
                    agent_name=agent_name,
                    dataset_name=dataset_name,
                    save=True,
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
