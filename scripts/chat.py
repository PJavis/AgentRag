#!/usr/bin/env python3
"""
PAM CLI Chat — interactive terminal client.

Usage:
    python scripts/chat.py [--url http://127.0.0.1:8000] [--doc <document_title>]

Commands (type inside the chat):
    /new [title]          Start a new conversation
    /list                 List recent conversations
    /switch <id_prefix>   Switch to an existing conversation
    /del                  Delete the current conversation
    /history [n]          Show last n messages (default 10)
    /doc [title]          Set or clear document filter
    /docs                 List ingested documents
    /help                 Show this help
    /exit  or  /quit      Exit

Ctrl+C during a request cancels it without exiting.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Any

import aiohttp

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"

def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

_RETRYABLE = (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, aiohttp.ClientConnectorError)

async def _post(session: aiohttp.ClientSession, url: str, body: dict, retries: int = 2) -> Any:
    for attempt in range(retries + 1):
        try:
            async with session.post(url, json=body) as r:
                if r.status >= 400:
                    text = await r.text()
                    raise RuntimeError(f"HTTP {r.status}: {text[:200]}")
                return await r.json()
        except _RETRYABLE:
            if attempt == retries:
                raise
            await asyncio.sleep(0.3)

async def _get(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.get(url) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"HTTP {r.status}: {text[:200]}")
        return await r.json()

async def _delete(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.delete(url) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"HTTP {r.status}: {text[:200]}")
        return await r.json()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_answer(text: str) -> str:
    """Basic markdown → terminal: bold **x**, inline code `x`."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: _c(m.group(1), BOLD), text)
    text = re.sub(r"`([^`]+)`",      lambda m: _c(m.group(1), CYAN), text)
    return text

def _fmt_conv(conv: dict, current_id: str | None = None) -> str:
    cid   = conv["conversation_id"]
    title = conv.get("title") or "(untitled)"
    ts    = (conv.get("created_at") or "")[:16].replace("T", " ")
    marker = _c(" ◀", CYAN) if cid == current_id else ""
    return f"  {_c(cid[:8], DIM)}  {_c(title, BOLD)}{marker}  {_c(ts, DIM)}"

def _fmt_msg(msg: dict) -> str:
    role    = msg.get("role", "?")
    content = (msg.get("content") or "").strip()
    ts      = (msg.get("created_at") or "")[:16].replace("T", " ")
    if role == "user":
        prefix = _c("You", YELLOW + BOLD)
    else:
        prefix = _c("PAM", GREEN + BOLD)
    return f"{_c(ts, DIM)} {prefix}: {content[:200]}"


# ── Main CLI ──────────────────────────────────────────────────────────────────

class PamCLI:
    def __init__(self, base_url: str, document_title: str | None):
        self.base      = base_url.rstrip("/")
        self.doc       = document_title
        self.conv_id: str | None = None
        self.conv_title: str | None = None

    def _prompt(self) -> str:
        doc_tag  = _c(f" [{self.doc}]", MAGENTA) if self.doc else ""
        conv_tag = _c(f" {self.conv_title or self.conv_id[:8] if self.conv_id else 'new'}", CYAN)
        return f"\n{_c('PAM', GREEN + BOLD)}{doc_tag}{conv_tag}{_c(' › ', BOLD)} "

    def _header(self) -> None:
        print(_c("=" * 60, DIM))
        print(_c("  PAM CLI Chat", BOLD + CYAN) + "  " + _c("type /help for commands", DIM))
        print(_c("=" * 60, DIM))

    async def _cmd_new(self, session: aiohttp.ClientSession, args: str) -> None:
        title = args.strip() or None
        data  = await _post(session, f"{self.base}/conversations",
                            {"title": title} if title else {})
        self.conv_id    = data["conversation_id"]
        self.conv_title = data.get("title")
        print(_c(f"✓ New conversation: {self.conv_id[:8]} — {self.conv_title or '(untitled)'}", GREEN))

    async def _cmd_list(self, session: aiohttp.ClientSession) -> None:
        data  = await _get(session, f"{self.base}/conversations?limit=20")
        convs = data.get("conversations", [])
        if not convs:
            print(_c("  No conversations yet.", DIM))
            return
        print(_c(f"\n  Recent conversations ({len(convs)}):", BOLD))
        for c in convs:
            print(_fmt_conv(c, self.conv_id))

    async def _cmd_switch(self, session: aiohttp.ClientSession, args: str) -> None:
        prefix = args.strip()
        if not prefix:
            print(_c("Usage: /switch <id_prefix>", YELLOW))
            return
        data  = await _get(session, f"{self.base}/conversations?limit=50")
        convs = data.get("conversations", [])
        match = [c for c in convs if c["conversation_id"].startswith(prefix)]
        if not match:
            print(_c(f"No conversation starting with '{prefix}'.", RED))
            return
        if len(match) > 1:
            print(_c("Ambiguous prefix — matches:", YELLOW))
            for c in match:
                print(_fmt_conv(c))
            return
        c = match[0]
        self.conv_id    = c["conversation_id"]
        self.conv_title = c.get("title")
        print(_c(f"✓ Switched to: {self.conv_id[:8]} — {self.conv_title or '(untitled)'}", GREEN))

    async def _cmd_del(self, session: aiohttp.ClientSession) -> None:
        if not self.conv_id:
            print(_c("No active conversation.", YELLOW))
            return
        confirm = input(_c(f"  Delete '{self.conv_title or self.conv_id[:8]}'? [y/N] ", YELLOW)).strip().lower()
        if confirm != "y":
            print(_c("  Cancelled.", DIM))
            return
        await _delete(session, f"{self.base}/conversations/{self.conv_id}")
        print(_c(f"✓ Deleted {self.conv_id[:8]}", RED))
        self.conv_id    = None
        self.conv_title = None

    async def _cmd_history(self, session: aiohttp.ClientSession, args: str) -> None:
        if not self.conv_id:
            print(_c("No active conversation.", YELLOW))
            return
        n = int(args.strip()) if args.strip().isdigit() else 10
        data = await _get(session, f"{self.base}/conversations/{self.conv_id}/messages?limit={n}")
        msgs = data.get("messages", [])
        if not msgs:
            print(_c("  No messages yet.", DIM))
            return
        print(_c(f"\n  Last {len(msgs)} messages:", BOLD))
        for m in msgs:
            print(_fmt_msg(m))

    async def _cmd_docs(self, session: aiohttp.ClientSession) -> None:
        try:
            data = await _get(session, f"{self.base}/documents")
        except RuntimeError:
            print(_c("  /documents endpoint not available.", DIM))
            return
        docs = data.get("documents", data) if isinstance(data, dict) else data
        if not docs:
            print(_c("  No documents ingested.", DIM))
            return
        print(_c(f"\n  Ingested documents ({len(docs)}):", BOLD))
        for d in docs:
            title = d.get("title") or d.get("source_id") or "?"
            print(f"  {_c(title, CYAN)}")

    def _cmd_doc(self, args: str) -> None:
        val = args.strip()
        if not val:
            self.doc = None
            print(_c("  Document filter cleared — searching all documents.", DIM))
        else:
            self.doc = val
            print(_c(f"✓ Document filter set to: {val}", GREEN))

    def _print_help(self) -> None:
        cmds = [
            ("/new [title]",       "Start a new conversation"),
            ("/list",              "List recent conversations"),
            ("/switch <id>",       "Switch to a conversation (by ID prefix)"),
            ("/del",               "Delete the current conversation"),
            ("/history [n]",       "Show last n messages (default 10)"),
            ("/doc [title]",       "Set / clear document filter"),
            ("/docs",              "List ingested documents"),
            ("/help",              "Show this help"),
            ("/exit | /quit",      "Exit"),
            ("Ctrl+C",             "Cancel the current request"),
        ]
        print(_c("\n  Commands:", BOLD))
        for cmd, desc in cmds:
            print(f"  {_c(cmd, CYAN):<28}{_c(desc, DIM)}")

    async def _send(self, session: aiohttp.ClientSession, question: str) -> None:
        body: dict[str, Any] = {"question": question}
        if self.conv_id:
            body["conversation_id"] = self.conv_id
        if self.doc:
            body["document_title"] = self.doc

        print(_c(f"\n  [{_ts()}] thinking…", DIM), end="", flush=True)
        try:
            data = await _post(session, f"{self.base}/chat", body)
        except asyncio.CancelledError:
            print(_c("\r  [cancelled]                   ", RED))
            return

        # Update conv_id from response
        if data.get("conversation_id"):
            self.conv_id = data["conversation_id"]

        answer = data.get("answer", "").strip()
        path   = data.get("reasoning_path", "semantic")
        total  = (data.get("timings_ms") or {}).get("total", 0)
        cits   = data.get("citations") or []

        print(f"\r  {_c('[' + _ts() + ']', DIM)} {_c(path, MAGENTA)} {_c(f'{total:.0f}ms', DIM)}")
        print()
        print(_c("PAM", GREEN + BOLD) + "  " + _fmt_answer(answer))

        if cits:
            print(_c(f"\n  Citations ({len(cits)}):", DIM))
            for i, c in enumerate(cits[:5], 1):
                sec = c.get("section_path") or ""
                print(_c(f"    [{i}] {sec}", DIM))

    async def run(self) -> None:
        self._header()
        if self.doc:
            print(_c(f"  Document filter: {self.doc}", MAGENTA))
        print(_c("  Type /help for commands, Ctrl+C to cancel a request.\n", DIM))

        # force_close=True: don't reuse TCP connections — avoids stale-connection
        # errors when the server closes keepalive between user keystrokes.
        connector = aiohttp.TCPConnector(limit=4, force_close=True)
        timeout   = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while True:
                try:
                    line = input(self._prompt()).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not line:
                    continue

                # ── Commands ─────────────────────────────────────────────
                if line.startswith("/"):
                    parts = line.split(maxsplit=1)
                    cmd   = parts[0].lower()
                    args  = parts[1] if len(parts) > 1 else ""

                    if cmd in ("/exit", "/quit"):
                        break
                    elif cmd == "/new":
                        await self._cmd_new(session, args)
                    elif cmd == "/list":
                        await self._cmd_list(session)
                    elif cmd == "/switch":
                        await self._cmd_switch(session, args)
                    elif cmd == "/del":
                        await self._cmd_del(session)
                    elif cmd == "/history":
                        await self._cmd_history(session, args)
                    elif cmd == "/doc":
                        self._cmd_doc(args)
                    elif cmd == "/docs":
                        await self._cmd_docs(session)
                    elif cmd == "/help":
                        self._print_help()
                    else:
                        print(_c(f"  Unknown command: {cmd}  (type /help)", YELLOW))
                    continue

                # ── Chat message ─────────────────────────────────────────
                task = asyncio.create_task(self._send(session, line))
                try:
                    await task
                except KeyboardInterrupt:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    print(_c("\r  [cancelled]                   ", RED))
                except Exception as exc:
                    print(_c(f"\n  Error: {exc}", RED))

        print(_c("\n  Goodbye.", DIM))


def main() -> None:
    parser = argparse.ArgumentParser(description="PAM CLI Chat")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="PAM server URL")
    parser.add_argument("--doc", default=None, help="Document title filter")
    args = parser.parse_args()

    cli = PamCLI(base_url=args.url, document_title=args.doc)
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print(_c("\n  Goodbye.", DIM))
        sys.exit(0)


if __name__ == "__main__":
    main()
