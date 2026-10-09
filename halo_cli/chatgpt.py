"""Companion CLI keeps the existing `halo TRACE -p PROMPT` interface intact."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import typer

from engine.chatgpt_auth import PLAN_SCOPE, RESOURCE, ChatGPTAuthError, CredentialStore
from engine.chatgpt_auth import login as sign_in
from engine.chatgpt_auth import logout as sign_out
from engine.chatgpt_client import ChatGPTClient, PlanTransport, plan_text
from engine.model_config import ModelConfig

cli = typer.Typer(add_completion=False, help="HALO-owned Sign in with ChatGPT sessions.")
_auth_file: Path | None = None


@cli.callback()
def configure(auth_file: Path | None = typer.Option(None, "--auth-file")) -> None:
    global _auth_file
    _auth_file = auth_file


@cli.command()
def login(
    new_account: bool = typer.Option(False, "--new-account"),
    enable_plan: bool = typer.Option(False, "--enable-plan"),
) -> None:
    """Continue with ChatGPT in the system browser (HALO-specific consent)."""
    typer.echo("Continue with ChatGPT in the browser. Waiting for HALO authorization...")
    issued = sign_in(CredentialStore(_auth_file), new_account=new_account, enable_plan=enable_plan)
    typer.echo(f"Saved ChatGPT account {issued}.")
    status()


@cli.command()
def status() -> None:
    """Show account labels and granted plan permission, never tokens."""
    store = CredentialStore(_auth_file)
    with store.lock:
        data = store.read()
    if not data["accounts"]:
        typer.echo("No saved ChatGPT accounts. Run halo-chatgpt login.")
    for issued, record in data["accounts"].items():
        active = "*" if issued == data.get("active") else " "
        plan = PLAN_SCOPE in record.get("scopes", [])
        signed_in = bool(record.get("id_token"))
        typer.echo(
            f"{active} {issued} | {record.get('email') or 'account'} | signed_in={signed_in} | plan_authorized={plan}"
        )


@cli.command("use")
def select(account: str) -> None:
    """Select one saved issued client ID; each workspace keeps its own tokens."""
    store = CredentialStore(_auth_file)
    with store.lock:
        data = store.read()
        store.selected(data, account)
        data["active"] = account
        store.write(data)
    typer.echo(f"Selected ChatGPT account {account}.")


@cli.command()
def logout() -> None:
    """Revoke the renewable session and clear this account's local tokens."""
    confirmed = sign_out(CredentialStore(_auth_file))
    typer.echo("Local tokens cleared; account registration retained.")
    if not confirmed:
        typer.echo(
            "Remote revocation was not confirmed. Disconnect HALO in ChatGPT Settings if needed."
        )


async def _models() -> None:
    async with httpx.AsyncClient(
        transport=PlanTransport(CredentialStore(_auth_file)), timeout=30
    ) as client:
        response = await client.get(f"{RESOURCE}/models")
        if response.is_error:
            raise ChatGPTAuthError(f"ChatGPT model catalog rejected (HTTP {response.status_code}).")
        models = response.json().get("models")
        if not isinstance(models, list):
            raise ChatGPTAuthError("Unexpected ChatGPT model catalog format.")
        for model in models:
            if model.get("visibility") == "list":
                typer.echo(f"{model['slug']}\t{model.get('display_name', model['slug'])}")


@cli.command()
def models() -> None:
    """Fetch models currently available to the selected account."""
    asyncio.run(_models())


async def _check(model: str) -> None:
    async with ChatGPTClient(CredentialStore(_auth_file)) as client:
        text = await plan_text(
            client, ModelConfig(name=model), "Reply briefly.", "Say exactly: HALO connected."
        )
        typer.echo(text)
        typer.echo("response.completed received.")


@cli.command()
def check(model: str = typer.Option(..., "--model")) -> None:
    """Make one small real inference request against ChatGPT plan usage."""
    asyncio.run(_check(model))


def app() -> None:
    try:
        cli()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    app()
