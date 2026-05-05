from __future__ import annotations

import time
from typing import Protocol

import jax

from scm.config import LogLevel


class SimLogger(Protocol):
    """Logger interface for `simulate`.

    All methods that receive JAX tracers (`on_outer_step`) must dispatch through
    `jax.debug.callback` so that `simulate` remains jittable.
    """

    def on_start(self) -> None: ...
    def on_outer_step(self, t, info: dict) -> None: ...
    def on_end(self) -> None: ...


class _SilentLogger:
    def on_start(self) -> None:
        pass

    def on_outer_step(self, t, info: dict) -> None:
        pass

    def on_end(self) -> None:
        pass


class _BoundaryLogger:
    def on_start(self) -> None:
        print("Begin simulation...")

    def on_outer_step(self, t, info: dict) -> None:
        pass

    def on_end(self) -> None:
        print("Simulation complete.")


def _format_long_duration(d: float) -> str:
    unit = "s"
    if d > 120:
        d /= 60
        unit = "min"
    if d > 120:
        d /= 60
        unit = "h"
    return f"{d:.1f}{unit}"


class _ProgressLogger:
    """Boundary messages plus per-outer-step timing, ETA, and adaptive info."""

    def __init__(self, n_total: int):
        self.n_total = n_total
        self.i = 0
        self.start_time: float | None = None
        self.last_time: float | None = None

    def on_start(self) -> None:
        print("Begin simulation...")

    def _host_callback(self, t, info: dict) -> None:
        current = time.time()
        perc = self.i / max(self.n_total - 1, 1) * 100
        extra = ""
        if "n_inner" in info:
            extra = f", inner steps: {int(info['n_inner'])}"
        if "dt_min" in info:
            extra += f", dt_min: {float(info['dt_min']):.3f}s"

        if self.last_time is None:
            self.start_time = current
            print(f"t={float(t):.0f} ({perc:.0f}%){extra}")
        else:
            duration = current - self.last_time
            eta = duration * (self.n_total - self.i)
            print(
                f"t={float(t):.0f} ({perc:.0f}%), this iter: {duration:.2f}s, "
                f"ETA: {_format_long_duration(eta)}{extra}"
            )
        self.last_time = current
        self.i += 1

    def on_outer_step(self, t, info: dict) -> None:
        jax.debug.callback(self._host_callback, t, info)

    def on_end(self) -> None:
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            print(f"Total elapsed time: {_format_long_duration(elapsed)}")
        print("Simulation complete.")


def make_logger(level: LogLevel, n_outer: int) -> SimLogger:
    if level == LogLevel.SILENT:
        return _SilentLogger()
    if level == LogLevel.BOUNDARY:
        return _BoundaryLogger()
    if level == LogLevel.PROGRESS:
        return _ProgressLogger(n_total=n_outer)
    raise ValueError(f"Unknown log level: {level}")
