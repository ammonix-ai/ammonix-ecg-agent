"""
QPSI Profiling Utilities — always-on microbenchmark toolkit.

Provides a ``benchmark()`` function inspired by R's ``microbenchmark`` package
that accepts named callables, runs them N times, and returns a timing comparison
table with ns/ms per call statistics.

Usage::

    from qpsi.profiling import benchmark

    results = benchmark(
        fit_1d=lambda: fit_gaussians_1d(t, mag, 2),
        fit_2d=lambda: fit_gaussians_2d(t, v2, 3),
        n=100,
    )
    print(results)  # TimingTable with min/median/mean/max per callable

No external profiling dependencies required (uses only ``time.perf_counter``).
For deeper profiling, see Part D notes on pyinstrument/scalene/line_profiler.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class TimingResult:
    """Timing statistics for a single benchmark target.

    All durations are stored in **seconds** internally. The ``ns``, ``us``,
    and ``ms`` properties provide convenient unit conversions for display.

    Attributes:
        name: Benchmark target name.
        times_s: Raw per-call durations in seconds.
        n_calls: Number of invocations.
        min_s: Fastest call (seconds).
        max_s: Slowest call (seconds).
        mean_s: Arithmetic mean (seconds).
        median_s: Median (seconds).
        stdev_s: Standard deviation (seconds); 0.0 when n < 2.
    """

    name: str
    times_s: tuple[float, ...]
    n_calls: int
    min_s: float
    max_s: float
    mean_s: float
    median_s: float
    stdev_s: float

    # ---- convenience unit accessors ----

    @property
    def min_ns(self) -> float:
        return self.min_s * 1e9

    @property
    def min_us(self) -> float:
        return self.min_s * 1e6

    @property
    def min_ms(self) -> float:
        return self.min_s * 1e3

    @property
    def median_ns(self) -> float:
        return self.median_s * 1e9

    @property
    def median_us(self) -> float:
        return self.median_s * 1e6

    @property
    def median_ms(self) -> float:
        return self.median_s * 1e3

    @property
    def mean_ns(self) -> float:
        return self.mean_s * 1e9

    @property
    def mean_us(self) -> float:
        return self.mean_s * 1e6

    @property
    def mean_ms(self) -> float:
        return self.mean_s * 1e3

    @property
    def max_ms(self) -> float:
        return self.max_s * 1e3


@dataclass
class TimingTable:
    """Collection of :class:`TimingResult` entries with a tabular ``__str__``.

    Attributes:
        results: Ordered list of timing results.
        unit: Display unit for the summary table (``"auto"``, ``"ns"``,
            ``"us"``, ``"ms"``, or ``"s"``).
    """

    results: List[TimingResult] = field(default_factory=list)
    unit: str = "auto"

    def _choose_unit(self) -> str:
        """Select the best display unit based on median of all results."""
        if self.unit != "auto":
            return self.unit
        if not self.results:
            return "ms"
        ref = statistics.median(r.median_s for r in self.results)
        if ref < 1e-6:
            return "ns"
        if ref < 1e-3:
            return "us"
        if ref < 1.0:
            return "ms"
        return "s"

    def _scale(self, unit: str) -> float:
        """Multiplier to convert seconds -> *unit*."""
        return {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}[unit]

    def __str__(self) -> str:
        if not self.results:
            return "(no benchmarks recorded)"
        unit = self._choose_unit()
        scale = self._scale(unit)

        # Column widths
        name_w = max(len(r.name) for r in self.results)
        header = (
            f"{'name':<{name_w}}  {'n':>6}  "
            f"{'min':>10}  {'median':>10}  {'mean':>10}  {'max':>10}  {'stdev':>10}  "
            f"unit"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for r in self.results:
            lines.append(
                f"{r.name:<{name_w}}  {r.n_calls:>6}  "
                f"{r.min_s * scale:>10.2f}  {r.median_s * scale:>10.2f}  "
                f"{r.mean_s * scale:>10.2f}  {r.max_s * scale:>10.2f}  "
                f"{r.stdev_s * scale:>10.2f}  "
                f"{unit}"
            )

        return "\n".join(lines)

    def as_dict(self) -> List[Dict[str, Any]]:
        """Return results as a list of plain dicts (for JSON serialization)."""
        return [
            {
                "name": r.name,
                "n_calls": r.n_calls,
                "min_ms": r.min_ms,
                "median_ms": r.median_ms,
                "mean_ms": r.mean_ms,
                "max_ms": r.max_ms,
                "stdev_ms": r.stdev_s * 1e3,
            }
            for r in self.results
        ]


def _run_single(
    fn: Callable[[], Any],
    n: int,
    warmup: int,
) -> List[float]:
    """Run *fn* with warmup then *n* timed iterations; return durations (s)."""
    # Warmup (results discarded)
    for _ in range(warmup):
        fn()

    times: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def benchmark(
    *,
    n: int = 100,
    warmup: int = 3,
    unit: str = "auto",
    **callables: Callable[[], Any],
) -> TimingTable:
    """Microbenchmark named callables and return a :class:`TimingTable`.

    Inspired by R's ``microbenchmark``: runs each callable *n* times after
    *warmup* warmup iterations and collects per-call timing statistics.

    Parameters
    ----------
    n : int
        Number of timed iterations per callable (default 100).
    warmup : int
        Number of untimed warmup iterations (default 3).
    unit : str
        Display unit: ``"auto"`` (default), ``"ns"``, ``"us"``, ``"ms"``,
        or ``"s"``.
    **callables
        Named callables to benchmark. Each must accept zero arguments.

    Returns
    -------
    TimingTable
        Table of :class:`TimingResult` entries with tabular ``__str__``.

    Examples
    --------
    >>> import numpy as np
    >>> t = benchmark(
    ...     zeros=lambda: np.zeros(1000),
    ...     ones=lambda: np.ones(1000),
    ...     n=50,
    ... )
    >>> print(t)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    table = TimingTable(unit=unit)

    for name, fn in callables.items():
        times = _run_single(fn, n, warmup)
        times_tuple = tuple(times)
        table.results.append(
            TimingResult(
                name=name,
                times_s=times_tuple,
                n_calls=n,
                min_s=min(times),
                max_s=max(times),
                mean_s=statistics.mean(times),
                median_s=statistics.median(times),
                stdev_s=statistics.stdev(times) if len(times) >= 2 else 0.0,
            )
        )

    return table


def time_block(label: str = "") -> "_TimingContext":
    """Context manager that records wall-clock time for a code block.

    Usage::

        with time_block("my_step") as tb:
            expensive_computation()
        print(f"{tb.label}: {tb.elapsed_ms:.2f} ms")

    Parameters
    ----------
    label : str
        Optional label for the timed block.

    Returns
    -------
    _TimingContext
        Context manager with ``elapsed_s`` and ``elapsed_ms`` attributes
        populated on exit.
    """
    return _TimingContext(label)


class _TimingContext:
    """Context manager for :func:`time_block`."""

    __slots__ = ("label", "elapsed_s", "elapsed_ms", "_t0")

    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed_s: float = 0.0
        self.elapsed_ms: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "_TimingContext":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_s = time.perf_counter() - self._t0
        self.elapsed_ms = self.elapsed_s * 1e3
