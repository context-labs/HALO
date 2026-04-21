"""typer CLI: ask a question about a codebase; Claude Agent answers with file:line citations."""
import asyncio
from pathlib import Path

import typer
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)
from dotenv import load_dotenv
from rich import print as rprint

from tracing import agent_span, setup_tracing


async def _run(question: str, root: Path, model: str) -> str:
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=(
            "You answer questions about a local codebase using the Read, Grep, and "
            "Glob tools. Start broad (Grep or Glob), then Read specific files. "
            "Answer concisely with file:line citations."
        ),
        allowed_tools=["Read", "Grep", "Glob"],
        cwd=str(root.resolve()),
    )

    last_text = ""
    with agent_span("claude-agent.run", model=model):
        async for message in query(prompt=question, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        last_text = block.text
    return last_text


def main(
    question: str = typer.Argument(..., help="Question about the codebase"),
    root: Path = typer.Option(Path("."), "--root", help="Directory the agent can read"),
    model: str = typer.Option("claude-sonnet-4-6", "--model", help="Claude model name"),
) -> None:
    load_dotenv()
    setup_tracing()
    answer = asyncio.run(_run(question, root, model))
    rprint(answer)


if __name__ == "__main__":
    typer.run(main)
