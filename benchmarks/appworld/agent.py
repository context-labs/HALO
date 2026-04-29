"""OpenAI Agents SDK adapter for the AppWorld benchmark.

A *thin* adapter: the agent has one tool, ``execute_code``, which forwards a
Python snippet to AppWorld's in-process IPython shell (``world.execute``).
This mirrors the paper's ReAct baseline (App. F of Trivedi et al., 2024) and
keeps every API invocation, stack trace, and supervisor lookup visible as a
TOOL span in the emitted trace.

We intentionally do *not* register one ``@function_tool`` per AppWorld API.
- Pros of one-tool: trivial schema, no need to predict allowed APIs, the
  full 457-API surface is reachable through ``apis.<app>.<api>(...)``.
- Cons: the agent has to know how to write Python, and tool descriptions
  cannot be optimized per-API. For HALO's first wired-up benchmark, the
  pros dominate — we want a working trace, not a leaderboard submission.
"""
from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, RunContextWrapper, function_tool
from appworld import AppWorld

# Default model. Override at the CLI; a small/cheap model is fine for traces.
# OpenAI Agents SDK defaults to the Responses API, so any model that supports
# Responses works here (gpt-4o-mini, gpt-4.1-mini, gpt-5, ...).
DEFAULT_MODEL = "gpt-4.1-mini"


@dataclass
class AgentContext:
    """Per-run context handing the live AppWorld ``world`` to tool callbacks.

    AppWorld's ``world.execute`` is stateful (Jupyter-style); the agent can
    reuse variables across turns, so we must keep the same ``AppWorld``
    instance alive for the duration of the run.
    """

    world: AppWorld


@function_tool
def execute_code(ctx: RunContextWrapper[AgentContext], code: str) -> str:
    """Execute a Python snippet in AppWorld's stateful IPython shell.

    The shell exposes:
    - ``apis.<app>.<api>(...)`` — call any of AppWorld's 457 APIs as a
      Python function. Auth tokens come from prior calls
      (``apis.<app>.login(...)`` returns ``access_token``); reuse them.
    - ``apis.api_docs.show_api_descriptions(app_name=...)`` — list APIs.
    - ``apis.api_docs.show_api_doc(app_name=..., api_name=...)`` — full
      docstring + arguments for a specific API.
    - ``apis.supervisor.<api>(...)`` — supervisor lookups (passwords,
      addresses, payment cards) and ``complete_task(answer=...)`` to end.

    Variables you ``print`` or assign persist across calls. Anything you
    don't ``print`` is invisible — the return value of this tool is the
    captured stdout (or a stack trace on failure).

    Args:
        code: A Python snippet, single line or multi-line.

    Returns:
        Captured stdout from the shell, or a traceback string if the snippet
        raised. AppWorld surfaces non-2xx API responses as exceptions, so
        bad arguments produce readable errors here.
    """
    return ctx.context.world.execute(code)


# Prompt mirrors the paper's "minimal ReAct" framing: short, no examples,
# just the essential affordances. HALO's optimizer is what re-writes prompts
# in production; for a first wired-up trace we want the baseline shape.
SYSTEM_PROMPT_TEMPLATE = """\
You are an autonomous coding agent helping a human supervisor on AppWorld, \
a simulated world of 9 day-to-day apps (Amazon, Spotify, Venmo, Gmail, \
Phone, SimpleNote, Splitwise, Todoist, FileSystem) plus 2 helper apps \
(api_docs, supervisor).

Supervisor:
- Name: {supervisor_first_name} {supervisor_last_name}
- Email: {supervisor_email}
- Phone: {supervisor_phone}

Task:
{instruction}

Available apps and one-line descriptions:
{app_descriptions}

Tool:
- ``execute_code(code: str)`` runs Python in a stateful IPython shell. Use \
``apis.<app>.<api>(...)`` to call APIs. Reuse variables across calls.

Workflow:
1. If unsure of an API's signature, look it up first with \
   ``apis.api_docs.show_api_descriptions(app_name='<app>')`` then \
   ``apis.api_docs.show_api_doc(app_name='<app>', api_name='<api>')``.
2. For protected apps, login first \
   (``response = apis.<app>.login(username=..., password=...)``) and reuse \
   the returned ``access_token`` in later calls.
3. When done, call ``apis.supervisor.complete_task(answer=<value>)`` if the \
   task asks for an answer, else ``apis.supervisor.complete_task()``.

Be concise. Do not narrate; just call the tool with the next snippet."""


def build_agent(model: str = DEFAULT_MODEL) -> Agent:
    """Build an Agent with a single ``execute_code`` tool.

    The system prompt is a placeholder template; per-task fields are filled
    in by ``run_one_task`` once the AppWorld task is loaded.
    """
    return Agent(
        name="HaloAppWorldAgent",
        instructions=SYSTEM_PROMPT_TEMPLATE,
        model=model,
        tools=[execute_code],
    )


def render_instructions(world: AppWorld) -> str:
    """Substitute task-specific fields into the system prompt template."""
    sup = world.task.supervisor
    descs = dict(world.task.app_descriptions)
    descs.pop("api_docs", None)
    descs.pop("supervisor", None)
    app_lines = "\n".join(f"- {name}: {desc}" for name, desc in descs.items())
    return SYSTEM_PROMPT_TEMPLATE.format(
        supervisor_first_name=sup["first_name"],
        supervisor_last_name=sup["last_name"],
        supervisor_email=sup["email"],
        supervisor_phone=sup["phone_number"],
        instruction=world.task.instruction,
        app_descriptions=app_lines,
    )
