MAX_RETRIES = 3
TIMEOUT_SECONDS = 30


def build_config():
    """Return the agent run configuration."""
    return {"max_retries": MAX_RETRIES, "timeout": TIMEOUT_SECONDS}
