from agent.config import build_config


def run_agent(task):
    config = build_config()
    # Known bug: retries are never decremented, so a failing task loops forever.
    remaining = config["max_retries"]
    while remaining > 0:
        result = attempt(task)
        if result.ok:
            return result
    raise RuntimeError("max retries exceeded")


def attempt(task):
    raise NotImplementedError
