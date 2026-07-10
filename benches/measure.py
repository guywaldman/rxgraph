"""Small, dependency-free measurement and reporting helpers."""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Measurement:
    name: str
    samples_seconds: list[float]
    result_size: int
    evaluated_edges: int = 0
    parallel_edges: int = 0
    execution: str = "serial"

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.samples_seconds)

    @property
    def p90_seconds(self) -> float:
        samples = sorted(self.samples_seconds)
        return samples[max(0, (len(samples) * 9 + 9) // 10 - 1)]


@dataclass(slots=True)
class TimedCase:
    name: str
    run: Callable[[], Any]
    normalize: Callable[[Any], Any]
    execution: str = "serial"
    stats: Callable[[Any], tuple[int, int]] = lambda _result: (0, 0)


def measure_cases(
    cases: list[TimedCase], *, warmups: int, runs: int
) -> list[Measurement]:
    """Preflight, warm up, then interleave samples without timing setup."""
    if not cases:
        return []
    expected = cases[0].normalize(cases[0].run())
    for case in cases:
        result = case.run()
        if case.normalize(result) != expected:
            raise RuntimeError(f"{case.name} returned a different result")
        _verify_execution(case, result)

    for _ in range(warmups):
        for case in cases:
            gc.collect()
            result = case.run()
            if case.normalize(result) != expected:
                raise RuntimeError(f"{case.name} changed result during warmup")
            _verify_execution(case, result)

    samples = {case.name: [] for case in cases}
    last: dict[str, Any] = {}
    for repeat in range(runs):
        order = cases[repeat:] + cases[:repeat]
        for case in order:
            gc.collect()
            started = time.perf_counter()
            result = case.run()
            samples[case.name].append(time.perf_counter() - started)
            if case.normalize(result) != expected:
                raise RuntimeError(f"{case.name} changed result while measuring")
            _verify_execution(case, result)
            last[case.name] = result

    return [
        Measurement(
            name=case.name,
            samples_seconds=samples[case.name],
            result_size=_result_size(case.normalize(last[case.name])),
            evaluated_edges=case.stats(last[case.name])[0],
            parallel_edges=case.stats(last[case.name])[1],
            execution=case.execution,
        )
        for case in cases
    ]


def measure_one(run: Callable[[], Any], *, warmups: int, runs: int) -> list[float]:
    for _ in range(warmups):
        gc.collect()
        run()
    samples = []
    for _ in range(runs):
        gc.collect()
        started = time.perf_counter()
        run()
        samples.append(time.perf_counter() - started)
    return samples


def as_json(
    measurement: Measurement, baseline_seconds: float | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": measurement.name,
        "samples_seconds": measurement.samples_seconds,
        "median_seconds": measurement.median_seconds,
        "p90_seconds": measurement.p90_seconds,
        "result_size": measurement.result_size,
        "evaluated_edges": measurement.evaluated_edges,
        "parallel_edges": measurement.parallel_edges,
        "execution": measurement.execution,
    }
    if baseline_seconds is not None:
        payload["baseline_speedup"] = baseline_seconds / measurement.median_seconds
    return payload


def _verify_execution(case: TimedCase, result: Any) -> None:
    _evaluated, parallel = case.stats(result)
    if case.execution == "parallel" and parallel == 0:
        raise RuntimeError(f"{case.name} silently fell back to serial execution")
    if case.execution == "serial" and parallel != 0:
        raise RuntimeError(f"{case.name} reported parallel work in serial execution")


def _result_size(result: Any) -> int:
    try:
        return len(result)
    except TypeError:
        return 1
