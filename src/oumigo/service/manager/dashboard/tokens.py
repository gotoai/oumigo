"""Token charts: vLLM's token counters turned into amounts per period.

A worker reports two cumulative counters every 5 s: ``vllm:generation_tokens_total``
(tokens generated) and ``vllm:prompt_tokens_total`` (input tokens processed). The
tokens in a period are the difference between consecutive samples, with one rule: a
counter that drops was reset by a vLLM restart, so the later value itself is the count
since the restart (never a negative amount). A sample that never arrived loses nothing:
the next one carries the total.

Units: MT = million tokens. Two views, each per worker and kind, plus an "All workers"
total per kind:

- **rate**, MT / min: the tokens of a slot or bucket divided by its duration;
- **cumulative**, MT: a running total from the chart's left edge.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from oumigo.service.manager.dashboard.aggregate import _trailing_moving_average

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
KINDS = {  # kind -> counter
    "generated": "vllm:generation_tokens_total",
    "input": "vllm:prompt_tokens_total",
}
COUNTER_KIND = {counter: kind for kind, counter in KINDS.items()}
MILLION = 1_000_000
RATE, CUMULATIVE = "rate", "cumulative"
_UNNAMED_SEQ = 10**9


def delta(previous: float | None, value: float) -> float:
    """Tokens since the previous sample; a drop means the counter restarted at 0."""
    if previous is None:
        return 0.0
    return value - previous if value >= previous else value


def _epoch(ts: str) -> int:
    return int(datetime.strptime(ts, TS_FORMAT).replace(tzinfo=timezone.utc).timestamp())


def cumulative(values: list[float | None]) -> list[float | None]:
    """Running total from the first non-empty point; flat across later gaps."""
    out: list[float | None] = []
    total: float | None = None
    for v in values:
        if v is not None:
            total = (total or 0.0) + v
        out.append(total)
    return out


def assemble(per_series: dict[tuple[str, str], list[float | None]],
             node_info: dict[str, dict] | None) -> list[dict]:
    """Chart series: per worker (ordered by Worker#N) and kind, then an "All workers"
    total per kind (the sum of the workers present at each point)."""
    node_info = node_info or {}

    def sort_key(node_id: str) -> tuple[int, str]:
        return ((node_info.get(node_id) or {}).get("seq") or _UNNAMED_SEQ, node_id)

    nodes = sorted({n for n, _ in per_series}, key=sort_key)
    series: list[dict] = []
    for node_id in nodes:
        worker = (node_info.get(node_id) or {}).get("name") or node_id[:8]
        for kind in KINDS:
            if (node_id, kind) in per_series:
                series.append({"node_id": node_id, "worker": worker, "kind": kind,
                               "total": False, "label": f"{worker} · {kind}",
                               "data": per_series[(node_id, kind)]})
    for kind in KINDS:
        rows = [per_series[(n, kind)] for n in nodes if (n, kind) in per_series]
        if not rows:
            continue
        summed = [None if all(r[i] is None for r in rows)
                  else round(sum(r[i] or 0.0 for r in rows), 6)
                  for i in range(len(rows[0]))]
        series.append({"node_id": None, "worker": "All workers", "kind": kind,
                       "total": True, "label": f"All workers · {kind}", "data": summed})
    return series


def live_series(rows: list[dict], now: datetime, mode: str,
                node_info: dict[str, dict] | None = None,
                window_s: int = 3600, grid_s: int = 5) -> dict:
    """The past `window_s` on the 5 s grid, from the console's raw mirror rows: rate
    (MT / min, with the GPU chart's 6-sample moving average) or cumulative (MT)."""
    samples: dict[tuple[str, str], dict[int, float]] = {}
    for r in rows:
        kind = COUNTER_KIND.get(r["metric"])
        if kind is not None:
            samples.setdefault((r["node_id"], kind), {})[_epoch(r["timestamp"])] = float(r["value"])

    end = math.floor(now.timestamp() / grid_s) * grid_s
    count = window_s // grid_s
    slots = [end - grid_s * (count - 1 - i) for i in range(count)]
    index = {s: i for i, s in enumerate(slots)}

    per_series: dict[tuple[str, str], list[float | None]] = {}
    for key, by_slot in samples.items():
        points: list[float | None] = [None] * count
        previous: tuple[int, float] | None = None
        for slot in sorted(by_slot):
            value = by_slot[slot]
            if previous is not None and slot in index:
                tokens = delta(previous[1], value)
                if mode == RATE:
                    minutes = (slot - previous[0]) / 60
                    points[index[slot]] = tokens / MILLION / minutes
                else:
                    points[index[slot]] = tokens / MILLION
            previous = (slot, value)
        if mode == RATE:
            per_series[key] = _trailing_moving_average(points, digits=6)
        else:
            per_series[key] = [None if v is None else round(v, 6) for v in cumulative(points)]
        if all(v is None for v in per_series[key]):
            del per_series[key]

    return {
        "generated_at": now.astimezone(timezone.utc).strftime(TS_FORMAT),
        "window_s": window_s,
        "grid_s": grid_s,
        "unit": "MT / min" if mode == RATE else "MT",
        "labels": [datetime.fromtimestamp(s, tz=timezone.utc).strftime("%H:%M:%S")
                   for s in slots],
        "series": assemble(per_series, node_info),
    }
