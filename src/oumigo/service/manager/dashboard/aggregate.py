"""Per-node GPU utilization series on the 5s grid (V1.0) — no temporal *rollup*.

The sheet keeps the worker's native 5-second grid (no minute bucketing). Two
combining steps: *spatial* — a node's several GPUs are averaged into one value per
slot (``average gpu:#N_util_pct of each node``); then a *trailing moving average*
smooths each series along the grid (see ``_trailing_moving_average``). Everything
here is pure and synchronous so it's trivially testable.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

TS_FORMAT = "%Y-%m-%d %H:%M:%S"  # UTC grid-slot stamp, matches the worker/store

# Per-GPU utilization gauge, e.g. "gpu:#0_util_pct". A node may have several.
_GPU_UTIL_RE = re.compile(r"^gpu:#(\d+)_util_pct$")

_UNNAMED_SEQ = 10**9  # sorts nodes without an assigned Worker#N after named ones

MA_WINDOW = 6  # trailing slots per moving-average window: [tN-5 .. tN]
MA_MIN_SAMPLES = 4  # windows with fewer present samples than this are left as gaps


def _trailing_moving_average(
    data: list[float | None], window: int = MA_WINDOW, min_samples: int = MA_MIN_SAMPLES
) -> list[float | None]:
    """Smooth a slot-aligned series with a trailing moving average.

    For slot ``i`` the window is the ``window`` slots ``[i-window+1 .. i]``; the
    average is over the *present* (non-``None``) samples in it. A window with fewer
    than ``min_samples`` present values stays ``None`` (a gap) — so the leading edge
    and sparse stretches don't produce misleading under-sampled averages.
    """
    out: list[float | None] = []
    for i in range(len(data)):
        present = [v for v in data[max(0, i - window + 1) : i + 1] if v is not None]
        out.append(round(sum(present) / len(present), 2) if len(present) >= min_samples else None)
    return out


def gpu_util_series(
    rows: list[dict],
    now: datetime,
    node_info: dict[str, dict] | None = None,
    window_s: int = 3600,
    grid_s: int = 5,
) -> dict:
    """Per-node average GPU utilization at each 5s grid slot over the last window.

    Averages ``gpu:#N_util_pct`` across a node's GPUs per slot, places that on the
    fixed 5s grid axis (aligned to the worker's ``:00/:05`` UTC slots), then smooths
    each series with a trailing moving average (see ``_trailing_moving_average``).
    Slots a node never reported stay ``None`` (a gap) — "workers visible time points"
    only. Series are labeled with the manager's ``Worker#N`` name and ordered by that
    sequence, so color follows the worker identity. Chart.js-friendly shape.
    """
    node_info = node_info or {}

    # Spatial mean: a node's GPUs -> one value per (node, slot).
    slot: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if not _GPU_UTIL_RE.match(r["metric"]):
            continue
        slot.setdefault((r["node_id"], r["timestamp"]), []).append(float(r["value"]))

    per_node: dict[str, dict[str, float]] = {}
    for (node_id, ts), vals in slot.items():
        per_node.setdefault(node_id, {})[ts] = sum(vals) / len(vals)

    # The fixed 5s grid axis, ending at the current (grid-aligned) slot.
    end_epoch = math.floor(now.timestamp() / grid_s) * grid_s
    count = window_s // grid_s
    slot_dts = [
        datetime.fromtimestamp(end_epoch - grid_s * (count - 1 - i), tz=timezone.utc)
        for i in range(count)
    ]
    slot_keys = [dt.strftime(TS_FORMAT) for dt in slot_dts]

    def sort_key(node_id: str) -> tuple[int, str]:
        info = node_info.get(node_id) or {}
        return (info.get("seq") or _UNNAMED_SEQ, node_id)

    series = []
    for node_id in sorted(per_node, key=sort_key):
        points = per_node[node_id]
        raw = [round(points[k], 2) if k in points else None for k in slot_keys]
        info = node_info.get(node_id) or {}
        series.append(
            {
                "node_id": node_id,
                "label": info.get("name") or node_id[:8],
                "data": _trailing_moving_average(raw),
            }
        )

    return {
        "generated_at": now.astimezone(timezone.utc).strftime(TS_FORMAT),
        "window_s": window_s,
        "grid_s": grid_s,
        "unit": "%",
        "labels": [dt.strftime("%H:%M:%S") for dt in slot_dts],
        "series": series,
    }
