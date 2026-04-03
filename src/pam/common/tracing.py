from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageEvent:
    stage: str
    service: str
    elapsed_ms: float
    metadata: dict[str, Any]
    error: str | None = None


class StageTracer:
    """
    Lightweight in-memory stage tracer cho observability.

    Ghi lại timing và metadata của từng stage trong một request.
    Thay thế dần các biến *_latency_ms rời rạc trong AgentService.
    Zero external dependency — chỉ dùng stdlib time.
    """

    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id
        self._events: list[StageEvent] = []
        self._starts: dict[str, tuple[float, str, dict[str, Any]]] = {}
        # stage → (start_time, service, start_metadata)

    @property
    def events(self) -> list[StageEvent]:
        return list(self._events)

    def start(self, stage: str, service: str, **metadata: Any) -> None:
        self._starts[stage] = (time.perf_counter(), service, metadata)

    def end(self, stage: str, **metadata: Any) -> StageEvent:
        started_at, service, start_meta = self._starts.pop(stage, (time.perf_counter(), "unknown", {}))
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        combined_meta = {**start_meta, **metadata}
        event = StageEvent(
            stage=stage,
            service=service,
            elapsed_ms=round(elapsed_ms, 2),
            metadata=combined_meta,
        )
        self._events.append(event)
        return event

    def fail(self, stage: str, error: Exception, **metadata: Any) -> StageEvent:
        started_at, service, start_meta = self._starts.pop(stage, (time.perf_counter(), "unknown", {}))
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        combined_meta = {**start_meta, **metadata}
        event = StageEvent(
            stage=stage,
            service=service,
            elapsed_ms=round(elapsed_ms, 2),
            metadata=combined_meta,
            error=f"{type(error).__name__}: {error}",
        )
        self._events.append(event)
        return event

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "stages": [
                {
                    "stage": e.stage,
                    "service": e.service,
                    "elapsed_ms": e.elapsed_ms,
                    "metadata": e.metadata,
                    **({"error": e.error} if e.error else {}),
                }
                for e in self._events
            ],
        }

    def as_timings_dict(self) -> dict[str, float]:
        """stage → elapsed_ms. Dùng cho response timings_ms field."""
        return {e.stage: e.elapsed_ms for e in self._events}

    def total_elapsed_ms(self) -> float:
        return sum(e.elapsed_ms for e in self._events)


if __name__ == "__main__":
    import asyncio

    tracer = StageTracer(request_id="test-001")

    tracer.start("retrieve", "KnowledgeService", query="test query")
    time.sleep(0.01)
    tracer.end("retrieve", chunk_count=5)

    tracer.start("schema_discovery", "SchemaDiscoveryModule")
    time.sleep(0.005)
    tracer.end("schema_discovery", tables=["product"])

    tracer.start("sql", "SQLReasoningEngine")
    try:
        raise ValueError("test error")
    except ValueError as e:
        tracer.fail("sql", e)

    result = tracer.as_dict()
    assert len(result["stages"]) == 3, f"Expected 3 stages, got {len(result['stages'])}"
    assert result["stages"][0]["stage"] == "retrieve"
    assert result["stages"][0]["metadata"]["chunk_count"] == 5
    assert result["stages"][2]["error"] is not None

    timings = tracer.as_timings_dict()
    assert "retrieve" in timings
    assert "schema_discovery" in timings
    assert "sql" in timings

    print("[OK] StageTracer smoke test passed")
    print(f"  stages: {[s['stage'] for s in result['stages']]}")
    print(f"  timings: {timings}")
    print(f"  total_elapsed_ms: {tracer.total_elapsed_ms():.2f}")
