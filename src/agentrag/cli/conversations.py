from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from src.agentrag.chat.history import ConversationStore
from src.agentrag.cli.state import (
    get_active_conversation,
    set_active_conversation,
    clear_active_conversation,
)

app = typer.Typer(help="Manage conversations.")
console = Console()


def _fmt_time(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        s = int(delta.total_seconds())
        if s < 60:
            return "just now"
        if s < 3600:
            return f"{s // 60}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"
    except Exception:
        return iso[:10]


def _short_id(cid: str) -> str:
    return cid[:8] if len(cid) > 8 else cid


@app.command("list")
def list_conversations(limit: int = typer.Option(20, "--limit", "-n", help="Max conversations to show")):
    """List all conversations."""
    asyncio.run(_list(limit))


async def _list(limit: int) -> None:
    store = ConversationStore()
    convs = await store.list_conversations(limit=limit)
    active_id, _ = get_active_conversation()

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    table.add_column("", width=2)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", min_width=20)
    table.add_column("Created", width=12)

    for c in convs:
        cid = c.get("conversation_id", "")
        title = c.get("title") or "[dim](no title)[/dim]"
        created = _fmt_time(c.get("created_at"))
        marker = "[green]✦[/green]" if cid == active_id else " "
        table.add_row(marker, _short_id(cid), title, created)

    if not convs:
        console.print("[dim]No conversations yet. Run [bold]agentrag chat[/bold] to start one.[/dim]")
        return

    console.print(table)
    console.print(f"[dim]{len(convs)} conversation(s). [green]✦[/green] = active[/dim]")


@app.command("new")
def new_conversation(
    title: str = typer.Argument(default="", help="Conversation title (optional)"),
):
    """Create a new conversation and set it as active."""
    asyncio.run(_new(title or None))


async def _new(title: str | None) -> None:
    store = ConversationStore()
    conv = await store.create_conversation(title=title)
    cid = conv["conversation_id"]
    set_active_conversation(cid, conv.get("title"))
    label = conv.get("title") or f"[dim]{_short_id(cid)}[/dim]"
    console.print(f"[green]✓[/green] Created conversation {label}")
    console.print(f"  [dim]ID: {cid}[/dim]")


@app.command("switch")
def switch_conversation(conversation_id: str = typer.Argument(..., help="Conversation ID (or prefix)")):
    """Switch active conversation."""
    asyncio.run(_switch(conversation_id))


async def _switch(prefix: str) -> None:
    store = ConversationStore()
    convs = await store.list_conversations(limit=100)
    matches = [c for c in convs if c.get("conversation_id", "").startswith(prefix)]

    if not matches:
        console.print(f"[red]✗[/red] No conversation found matching [bold]{prefix}[/bold]")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[red]✗[/red] Ambiguous prefix — {len(matches)} matches. Use more characters.")
        raise typer.Exit(1)

    conv = matches[0]
    cid = conv["conversation_id"]
    set_active_conversation(cid, conv.get("title"))
    label = conv.get("title") or _short_id(cid)
    console.print(f"[green]✓[/green] Switched to [bold]{label}[/bold]  [dim]{cid}[/dim]")


@app.command("delete")
def delete_conversation(
    conversation_id: str = typer.Argument(..., help="Conversation ID (or prefix)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a conversation."""
    asyncio.run(_delete(conversation_id, yes))


async def _delete(prefix: str, yes: bool) -> None:
    store = ConversationStore()
    convs = await store.list_conversations(limit=100)
    matches = [c for c in convs if c.get("conversation_id", "").startswith(prefix)]

    if not matches:
        console.print(f"[red]✗[/red] No conversation found matching [bold]{prefix}[/bold]")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[red]✗[/red] Ambiguous prefix — {len(matches)} matches. Use more characters.")
        raise typer.Exit(1)

    conv = matches[0]
    cid = conv["conversation_id"]
    label = conv.get("title") or _short_id(cid)

    if not yes:
        typer.confirm(f"Delete conversation '{label}'?", abort=True)

    deleted = await store.delete_conversation(cid)
    if not deleted:
        console.print(f"[red]✗[/red] Could not delete conversation {cid}")
        raise typer.Exit(1)

    active_id, _ = get_active_conversation()
    if active_id == cid:
        clear_active_conversation()
        console.print("[dim]Active conversation cleared.[/dim]")

    console.print(f"[green]✓[/green] Deleted [bold]{label}[/bold]")


@app.command("show")
def show_active():
    """Show the currently active conversation."""
    active_id, active_title = get_active_conversation()
    if not active_id:
        console.print("[dim]No active conversation. Run [bold]agentrag conversations new[/bold] or [bold]agentrag chat[/bold].[/dim]")
        return
    label = active_title or "(no title)"
    console.print(f"[green]✦[/green] [bold]{label}[/bold]")
    console.print(f"  [dim]{active_id}[/dim]")
