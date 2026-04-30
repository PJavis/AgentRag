#!/usr/bin/env python3
"""ARQ worker auto-scaler.

Polls the ARQ Redis queue depth and spawns/kills worker processes to match
demand.  One scaler process manages N worker processes; workers are ordinary
`arq` CLI processes that can be inspected with standard ARQ tooling.

Configuration via environment variables (all optional):
  SCALER_MIN_WORKERS      Minimum live workers          (default: 1)
  SCALER_MAX_WORKERS      Maximum live workers          (default: 4)
  SCALER_SCALE_UP_AT      Queue depth per extra worker  (default: 5)
  SCALER_POLL_SECONDS     Seconds between polls         (default: 5)
  SCALER_COOLDOWN_SECONDS Min seconds between rescales  (default: 30)

Usage:
  python scaler.py

Stop:
  Ctrl-C  or  kill <pid>   (SIGTERM)  →  graceful drain of all workers
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [scaler] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scaler")

# ── Config ────────────────────────────────────────────────────────────────────
MIN_WORKERS      = int(os.getenv("SCALER_MIN_WORKERS", "1"))
MAX_WORKERS      = int(os.getenv("SCALER_MAX_WORKERS", "4"))
SCALE_UP_AT      = int(os.getenv("SCALER_SCALE_UP_AT", "5"))
POLL_SECONDS     = float(os.getenv("SCALER_POLL_SECONDS", "5"))
COOLDOWN_SECONDS = float(os.getenv("SCALER_COOLDOWN_SECONDS", "30"))

ARQ_QUEUE_KEY = "arq:queue"  # ARQ default sorted-set key in Redis

WORKER_CMD = [
    sys.executable, "-m", "arq",
    "src.agentrag.worker.settings.WorkerSettings",
]


# ── Queue depth ───────────────────────────────────────────────────────────────

async def get_queue_depth(redis_url: str) -> int:
    """Return number of pending (not yet started) jobs in the ARQ queue."""
    import redis.asyncio as aioredis  # already installed via arq dependency
    client = aioredis.from_url(redis_url, decode_responses=False)
    try:
        return int(await client.zcard(ARQ_QUEUE_KEY))
    except Exception as exc:
        logger.warning("Redis read failed: %s", exc)
        return 0
    finally:
        await client.aclose()


# ── Worker pool ───────────────────────────────────────────────────────────────

class WorkerPool:
    def __init__(self) -> None:
        self._procs: list[subprocess.Popen] = []
        self._last_scale: float = 0.0

    # ── internals ──────────────────────────────────────────────────────────

    def _reap(self) -> None:
        alive, dead = [], []
        for p in self._procs:
            (alive if p.poll() is None else dead).append(p)
        for p in dead:
            logger.info("Worker PID %d exited (rc=%d)", p.pid, p.returncode)
        self._procs = alive

    def _in_cooldown(self) -> bool:
        return time.monotonic() - self._last_scale < COOLDOWN_SECONDS

    def _spawn(self, *, bypass_cooldown: bool = False) -> None:
        if not bypass_cooldown and self._in_cooldown():
            return
        proc = subprocess.Popen(WORKER_CMD)
        self._procs.append(proc)
        self._last_scale = time.monotonic()
        logger.info("Spawned worker PID %d (total workers: %d)", proc.pid, len(self._procs))

    def _stop_one(self) -> None:
        if not self._procs or self._in_cooldown():
            return
        proc = self._procs.pop()
        proc.send_signal(signal.SIGTERM)
        self._last_scale = time.monotonic()
        logger.info("Sent SIGTERM to PID %d (total workers: %d)", proc.pid, len(self._procs))

    # ── public ─────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._procs)

    def ensure_minimum(self) -> None:
        """Spawn until at least MIN_WORKERS are alive (bypasses cooldown)."""
        self._reap()
        while len(self._procs) < MIN_WORKERS:
            self._spawn(bypass_cooldown=True)
            time.sleep(0.1)  # stagger startups

    def adjust(self, depth: int) -> None:
        """Scale workers up or down by one step based on queue depth."""
        self._reap()

        # desired = MIN + one extra worker per SCALE_UP_AT jobs
        desired = min(MAX_WORKERS, MIN_WORKERS + math.floor(depth / SCALE_UP_AT))
        current = self.count

        if current == desired:
            return

        if current < desired:
            logger.info(
                "Queue depth %d → scale up %d→%d workers",
                depth, current, current + 1,
            )
            self._spawn()
        elif current > desired and current > MIN_WORKERS:
            logger.info(
                "Queue depth %d → scale down %d→%d workers",
                depth, current, current - 1,
            )
            self._stop_one()

        # Safety net: never drop below minimum after reap
        self.ensure_minimum()

    def stop_all(self) -> None:
        """SIGTERM every worker; SIGKILL stragglers after 15 s."""
        if not self._procs:
            return
        pids = [p.pid for p in self._procs]
        for proc in self._procs:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
        logger.info("Waiting for workers %s to drain…", pids)
        deadline = time.monotonic() + 15
        for proc in self._procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Worker PID %d did not exit — sending SIGKILL", proc.pid)
                proc.kill()
        self._procs.clear()
        logger.info("All workers stopped")


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run(redis_url: str) -> None:
    pool = WorkerPool()
    shutdown = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: (
                logger.info("Received signal %d — initiating shutdown", s),
                shutdown.set(),
            ),
        )

    logger.info(
        "Scaler ready  min=%d  max=%d  scale_up_at=%d  cooldown=%ds  poll=%ds",
        MIN_WORKERS, MAX_WORKERS, SCALE_UP_AT, int(COOLDOWN_SECONDS), int(POLL_SECONDS),
    )
    pool.ensure_minimum()

    while not shutdown.is_set():
        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown.wait()),
                timeout=POLL_SECONDS,
            )
        except asyncio.TimeoutError:
            pass

        if shutdown.is_set():
            break

        depth = await get_queue_depth(redis_url)
        logger.debug("Queue depth=%d  workers=%d", depth, pool.count)
        pool.adjust(depth)

    pool.stop_all()


def main() -> None:
    try:
        from src.agentrag.config import settings
        redis_url = settings.REDIS_URL or "redis://127.0.0.1:6379/0"
    except Exception:
        redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    asyncio.run(run(redis_url))


if __name__ == "__main__":
    main()
