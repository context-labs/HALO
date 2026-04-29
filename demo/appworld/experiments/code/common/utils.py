import os

from appworld import load_task_ids
from appworld.cli import extract_dataset_name
from appworld.common.path_store import path_store
from appworld.common.text import render_template
from appworld_agents.code.common.usage_tracker import Usage


def fill_model_server_url(base_url: str) -> str:
    arguments = {"MODEL_SERVER_URL": os.environ["MODEL_SERVER_URL"]}
    return render_template(base_url, allow_missing=False, skip_fields=None, **arguments)


def get_overall_usage(experiment_name: str, skip_missing: bool = False) -> Usage:
    dataset_name = extract_dataset_name(experiment_name)
    task_ids = load_task_ids(dataset_name)
    overall_usage = Usage()
    for task_id in task_ids:
        file_path = os.path.join(
            path_store.experiment_outputs,
            experiment_name,
            "tasks",
            task_id,
            "misc",
            "usage.json",
        )
        if skip_missing and not os.path.exists(file_path):
            continue
        task_usage = Usage.load(file_path)
        overall_usage = overall_usage + task_usage
    return overall_usage
