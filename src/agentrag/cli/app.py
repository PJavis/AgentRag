from __future__ import annotations

import typer

from src.agentrag.cli.chat import chat
from src.agentrag.cli import conversations as conversations_module

cli_app = typer.Typer(
    name="agentrag",
    help="AgentRag — interactive RAG agent CLI.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

cli_app.command("chat")(chat)
cli_app.add_typer(conversations_module.app, name="conversations", help="Manage conversations.")


def main() -> None:
    cli_app()
