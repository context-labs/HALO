"""typer CLI: ask a question about a codebase; agent answers with file:line citations."""
from pathlib import Path

import typer
from agents import Runner
from dotenv import load_dotenv
from rich import print as rprint

from agent import AgentContext, build_agent
from tracing import setup_tracing


def main(
    question: str = typer.Argument(..., help="Question about the codebase"),
    root: Path = typer.Option(Path("."), "--root", help="Directory the agent can read"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="OpenAI model name"),
) -> None:
    load_dotenv()
    processor = setup_tracing()
    agent = build_agent(model)
    ctx = AgentContext(root=root.resolve())
    result = Runner.run_sync(agent, question, context=ctx)
    rprint(result.final_output)
    processor.shutdown()


if __name__ == "__main__":
    typer.run(main)
