from __future__ import annotations

import asyncio
import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from src.agentrag.agent.service import AgentService
from src.agentrag.chat.history import ConversationStore
from src.agentrag.cli.state import (
    get_active_conversation,
    set_active_conversation,
)
from src.agentrag.config import settings

console = Console()
err_console = Console(stderr=True, style="red")

_HELP_TEXT = (
    "[dim]Commands: /new [title]  /switch <id>  /list  /clear  exit  quit[/dim]"
)


def _parse_sse(chunk: str) -> tuple[str, dict]:
    event, data = "message", {}
    for line in chunk.strip().splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:].strip())
            except Exception:
                pass
    return event, data


def _print_citations(citations: list[dict]) -> None:
    if not citations:
        return
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("", style="dim", width=4)
    table.add_column("", style="dim")
    for i, c in enumerate(citations, 1):
        doc = c.get("document_title") or "—"
        section = c.get("section_path") or ""
        score = c.get("score")
        score_str = f"  [dim]{score:.2f}[/dim]" if score else ""
        ref = f"{doc}  §  {section}{score_str}" if section else f"{doc}{score_str}"
        table.add_row(f"[{i}]", ref)
    console.print(table)


async def _stream_turn(
    agent: AgentService,
    question: str,
    document_title: str | None,
    history: list[dict],
) -> tuple[str, list[dict], dict]:
    """Stream one turn. Returns (full_answer, citations, done_data)."""
    tokens: list[str] = []
    citations: list[dict] = []
    done_data: dict = {}
    answer_started = False

    status_spinner = Spinner("dots", text=" ")
    status_text = Text()

    with Live(status_spinner, console=console, refresh_per_second=12, transient=True):
        async for chunk in agent.chat_stream(
            question=question,
            document_title=document_title,
            chat_history=history,
        ):
            event, data = _parse_sse(chunk)

            if event == "status":
                step = data.get("step", "")
                status_spinner.update(text=f"[dim] {step}…[/dim]")

            elif event == "token":
                token = data.get("text", "")
                if token:
                    tokens.append(token)
                    if not answer_started:
                        answer_started = True
                    status_text.append(token)
                    status_spinner.update(text=status_text)

            elif event == "done":
                done_data = data
                citations = data.get("citations") or []

            elif event == "error":
                err_console.print(f"\n[red]Error:[/red] {data.get('message', 'unknown error')}")

    return "".join(tokens), citations, done_data


async def _chat_loop(
    conversation_id: str | None,
    document_title: str | None,
) -> None:
    store = ConversationStore()

    # Resolve or create conversation
    active_id, active_title = get_active_conversation()
    resolved_id = conversation_id or active_id

    conv = await store.get_or_create_conversation(conversation_id=resolved_id, title=None)
    cid = conv["conversation_id"]
    title = conv.get("title") or "(no title)"
    set_active_conversation(cid, conv.get("title"))

    # Header
    console.print(Panel(
        f"[bold]AgentRag[/bold]  ·  [dim]{title}[/dim]\n{_HELP_TEXT}",
        border_style="dim",
        padding=(0, 1),
    ))

    agent = AgentService()

    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "/exit", "/quit"):
            console.print("[dim]Bye.[/dim]")
            break

        # Inline commands
        if question.startswith("/new"):
            parts = question.split(maxsplit=1)
            new_title = parts[1] if len(parts) > 1 else None
            conv = await store.create_conversation(title=new_title)
            cid = conv["conversation_id"]
            title = conv.get("title") or "(no title)"
            set_active_conversation(cid, conv.get("title"))
            console.print(f"[green]✓[/green] New conversation: [bold]{title}[/bold]  [dim]{cid}[/dim]")
            agent = AgentService()
            continue

        if question.startswith("/switch"):
            parts = question.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[dim]Usage: /switch <conversation-id-prefix>[/dim]")
                continue
            prefix = parts[1].strip()
            convs = await store.list_conversations(limit=100)
            matches = [c for c in convs if c.get("conversation_id", "").startswith(prefix)]
            if not matches:
                console.print(f"[red]No conversation matching '{prefix}'[/red]")
                continue
            if len(matches) > 1:
                console.print(f"[red]Ambiguous prefix — {len(matches)} matches[/red]")
                continue
            conv = matches[0]
            cid = conv["conversation_id"]
            title = conv.get("title") or "(no title)"
            set_active_conversation(cid, conv.get("title"))
            console.print(f"[green]✓[/green] Switched to [bold]{title}[/bold]  [dim]{cid}[/dim]")
            agent = AgentService()
            continue

        if question == "/list":
            convs = await store.list_conversations(limit=20)
            for c in convs:
                marker = "[green]✦[/green]" if c.get("conversation_id") == cid else " "
                label = c.get("title") or "(no title)"
                short = c.get("conversation_id", "")[:8]
                console.print(f"  {marker} [dim]{short}[/dim]  {label}")
            continue

        if question == "/clear":
            conv = await store.create_conversation(title=None)
            cid = conv["conversation_id"]
            set_active_conversation(cid, None)
            console.print("[dim]Started fresh conversation.[/dim]")
            agent = AgentService()
            continue

        # Save user message
        await store.append_message(
            conversation_id=cid,
            role="user",
            content=question,
            extra_metadata={"document_title": document_title},
        )

        history = await store.list_messages(
            conversation_id=cid,
            limit=settings.CHAT_HISTORY_WINDOW,
        )

        # Stream answer
        console.print("[bold green]Assistant:[/bold green]")
        answer, citations, done_data = await _stream_turn(
            agent, question, document_title, history
        )

        if answer:
            console.print(Markdown(answer))
        else:
            console.print("[dim](no answer)[/dim]")

        _print_citations(citations)

        reasoning = done_data.get("reasoning_path")
        sql = done_data.get("sql_query")
        if sql:
            console.print(f"[dim]SQL: {sql}[/dim]")
        if reasoning and reasoning != "semantic":
            console.print(f"[dim]path: {reasoning}[/dim]")

        console.print()

        # Save assistant message
        if answer:
            await store.append_message(
                conversation_id=cid,
                role="assistant",
                content=answer,
                citations=citations,
                extra_metadata={"document_title": document_title},
            )


def chat(
    new: Annotated[bool, typer.Option("--new", "-n", help="Start a new conversation")] = False,
    title: Annotated[str, typer.Option("--title", "-t", help="Title for new conversation")] = "",
    document: Annotated[str, typer.Option("--document", "-d", help="Scope to a specific document title")] = "",
    conversation_id: Annotated[str, typer.Option("--id", help="Resume a specific conversation by ID")] = "",
):
    """Start an interactive chat session."""
    target_id: str | None = conversation_id or None
    doc: str | None = document or None

    if new:
        async def _start_new() -> str:
            store = ConversationStore()
            conv = await store.create_conversation(title=title or None)
            cid = conv["conversation_id"]
            set_active_conversation(cid, conv.get("title"))
            return cid
        target_id = asyncio.run(_start_new())

    asyncio.run(_chat_loop(conversation_id=target_id, document_title=doc))
