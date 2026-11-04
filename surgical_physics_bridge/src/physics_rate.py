"""
Physics-loop rate instrumentation.

``PhysicsRateMonitor`` records, for every physics tick, the period since the previous tick start
and the compute time of the tick itself, in fixed-size ring buffers. The buffers can live in a
``multiprocessing.RawArray`` so that the physics process writes them and the streaming process
reads them without any lock on the 1 kHz path (a reader may see a partially updated ring, which
only perturbs statistics, never physics).

Metrics (over the most recent ``window`` ticks):
    achieved_hz      -- ticks / sum(periods)
    period_p50_ms    -- median tick period
    period_p99_ms    -- 99th percentile tick period
    period_max_ms    -- worst tick period in the window
    compute_p99_ms   -- 99th percentile time spent inside the tick body
    overruns         -- cumulative ticks whose period exceeded ``overrun_factor * target_dt``
    resyncs          -- cumulative times the loop fell > ``max_lag_ticks`` behind and skipped deadlines
    total_ticks      -- cumulative ticks
"""

import multiprocessing as mp
from typing import Dict, Optional

import numpy as np

_HDR = 6  # count, overruns, target_dt, window, resyncs, overrun_factor


class PhysicsRateMonitor:
    def __init__(
        self,
        target_hz: float = 1000.0,
        window: int = 5000,
        overrun_factor: float = 1.5,
        shared_buffer=None,
    ):
        size = _HDR + 2 * window
        if shared_buffer is None:
            self._buf = np.zeros(size, dtype=np.float64)
        else:
            self._buf = np.frombuffer(shared_buffer, dtype=np.float64)
            if self._buf.size != size:
                raise ValueError(f"shared buffer has {self._buf.size} doubles, expected {size}")
        self._buf[2] = 1.0 / target_hz
        self._buf[3] = window
        self._buf[5] = overrun_factor
        self.window = window
        self._periods = self._buf[_HDR:_HDR + window]
        self._compute = self._buf[_HDR + window:]
        self._last_start: Optional[float] = None
        self._overrun_threshold = overrun_factor / target_hz

    @staticmethod
    def allocate_shared(window: int = 5000):
        """Allocates a lock-free shared buffer usable across processes (pass to ``shared_buffer``)."""
        return mp.get_context("spawn").RawArray("d", _HDR + 2 * window)

    @classmethod
    def reader(cls, shared_buffer) -> "PhysicsRateMonitor":
        """Read-only view on a buffer written by another process (keeps the writer's settings)."""
        buf = np.frombuffer(shared_buffer, dtype=np.float64)
        window = int(buf[3])
        obj = cls.__new__(cls)
        obj._buf = buf
        obj.window = window
        obj._periods = buf[_HDR:_HDR + window]
        obj._compute = buf[_HDR + window:]
        obj._last_start = None
        obj._overrun_threshold = buf[5] * buf[2]
        return obj

    def tick(self, t_start: float, t_end: float) -> None:
        """Record one tick that started at ``t_start`` and finished its work at ``t_end`` (seconds)."""
        buf = self._buf
        if self._last_start is not None:
            period = t_start - self._last_start
            i = int(buf[0]) % self.window
            self._periods[i] = period
            self._compute[i] = t_end - t_start
            buf[0] += 1.0
            if period > self._overrun_threshold:
                buf[1] += 1.0
        self._last_start = t_start

    def note_resync(self) -> None:
        self._buf[4] += 1.0

    def reset(self) -> None:
        self._buf[0] = 0.0
        self._buf[1] = 0.0
        self._buf[4] = 0.0
        self._last_start = None

    def snapshot(self) -> Dict[str, float]:
        count = int(self._buf[0])
        n = min(count, self.window)
        out = {
            "total_ticks": count,
            "overruns": int(self._buf[1]),
            "resyncs": int(self._buf[4]),
            "target_hz": 1.0 / self._buf[2] if self._buf[2] > 0 else float("nan"),
        }
        if n == 0:
            out.update(achieved_hz=0.0, period_p50_ms=float("nan"), period_p99_ms=float("nan"),
                       period_max_ms=float("nan"), compute_p99_ms=float("nan"))
            return out
        per = np.array(self._periods[:n]) * 1e3
        comp = np.array(self._compute[:n]) * 1e3
        total = per.sum()
        out.update(
            achieved_hz=float(n / (total * 1e-3)) if total > 0 else float("nan"),
            period_p50_ms=float(np.percentile(per, 50)),
            period_p99_ms=float(np.percentile(per, 99)),
            period_max_ms=float(per.max()),
            compute_p99_ms=float(np.percentile(comp, 99)),
        )
        return out

    @staticmethod
    def format(stats: Dict[str, float]) -> str:
        return (
            f"physics {stats['achieved_hz']:.1f} Hz (target {stats['target_hz']:.0f}) | "
            f"period p50 {stats['period_p50_ms']:.3f} ms, p99 {stats['period_p99_ms']:.3f} ms, "
            f"max {stats['period_max_ms']:.2f} ms | tick compute p99 {stats['compute_p99_ms']:.3f} ms | "
            f"overruns {stats['overruns']}/{stats['total_ticks']} | resyncs {stats['resyncs']}"
        )
