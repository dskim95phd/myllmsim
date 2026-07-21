import json
import os
import hashlib
import tracemalloc
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter


_active_recorder = ContextVar("llmservingsim_host_timing_recorder", default=None)


def _percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _process_memory_kib(pid):
    """Read current and peak RSS from Linux procfs when available."""
    if pid is None:
        return None
    try:
        values = {}
        with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as status:
            for line in status:
                key, separator, raw_value = line.partition(":")
                if not separator or key not in ("VmRSS", "VmHWM"):
                    continue
                fields = raw_value.split()
                if fields:
                    values[key] = int(fields[0])
        if not values:
            return None
        return {
            "pid": int(pid),
            "rss_kib": values.get("VmRSS"),
            "hwm_kib": values.get("VmHWM"),
        }
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None


class HostTimingRecorder:
    """Collect host-side stage durations and transport counters."""

    def __init__(self):
        self._started_at = perf_counter()
        self._samples = defaultdict(list)
        self._values = defaultdict(list)
        self._counters = Counter()
        self._digests = {}
        self._digest_counts = Counter()
        self._rss_checkpoints = []

    @contextmanager
    def measure(self, stage):
        start = perf_counter()
        try:
            yield
        finally:
            self.record(stage, perf_counter() - start)

    def record(self, stage, duration_seconds):
        self._samples[stage].append(float(duration_seconds))

    def increment(self, name, amount=1):
        self._counters[name] += amount

    def observe(self, name, value):
        """Record a dimensionless workload value such as batch size."""
        self._values[name].append(float(value))

    def update_digest(self, name, payload):
        """Append a length-delimited byte record to a deterministic digest."""
        digest = self._digests.setdefault(name, hashlib.sha256())
        payload = bytes(payload)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        self._digest_counts[name] += 1

    def record_rss_checkpoint(self, name, backend_pid=None, metadata=None):
        """Capture current process RSS at a named lifecycle checkpoint."""
        checkpoint = {
            "name": str(name),
            "elapsed_seconds": perf_counter() - self._started_at,
            "frontend": _process_memory_kib(os.getpid()),
            "backend": _process_memory_kib(backend_pid),
        }
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            checkpoint["python_traced_memory"] = {
                "current_bytes": current,
                "peak_bytes": peak,
            }
        if metadata:
            checkpoint["metadata"] = dict(metadata)
        self._rss_checkpoints.append(checkpoint)
        return checkpoint

    def sample_count(self, stage):
        return len(self._samples.get(stage, ()))

    def summary(self, metadata=None):
        stages = {}
        for stage, values in sorted(self._samples.items()):
            total = sum(values)
            warm = values[1:]
            stages[stage] = {
                "count": len(values),
                "total_seconds": total,
                "mean_seconds": total / len(values),
                "p50_seconds": _percentile(values, 0.50),
                "p90_seconds": _percentile(values, 0.90),
                "p99_seconds": _percentile(values, 0.99),
                "max_seconds": max(values),
                "cold_first_seconds": values[0],
                "warm_count": len(warm),
                "warm_total_seconds": sum(warm),
                "warm_p50_seconds": _percentile(warm, 0.50),
                "warm_p90_seconds": _percentile(warm, 0.90),
                "warm_p99_seconds": _percentile(warm, 0.99),
                "warm_max_seconds": max(warm) if warm else 0.0,
            }
        metrics = {}
        for metric, values in sorted(self._values.items()):
            metrics[metric] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "p50": _percentile(values, 0.50),
                "p90": _percentile(values, 0.90),
                "p99": _percentile(values, 0.99),
                "max": max(values),
            }
        result = {
            "schema_version": 2,
            "metadata": dict(metadata or {}),
            "stages": stages,
            "metrics": metrics,
            "rss_checkpoints": list(self._rss_checkpoints),
            "counters": dict(sorted(self._counters.items())),
            "digests": {
                name: {
                    "count": self._digest_counts[name],
                    "sha256": digest.hexdigest(),
                }
                for name, digest in sorted(self._digests.items())
            },
        }
        if tracemalloc.is_tracing():
            result["tracemalloc_top"] = [
                {
                    "location": str(statistic.traceback),
                    "size_bytes": statistic.size,
                    "count": statistic.count,
                }
                for statistic in tracemalloc.take_snapshot().statistics(
                    "traceback")[:25]
            ]
        return result

    def write_json(self, path, metadata=None):
        path = os.path.abspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as output:
            json.dump(self.summary(metadata), output, indent=2, sort_keys=True)
            output.write("\n")


def current_recorder():
    """Return the recorder active in the current trace-generation call."""

    return _active_recorder.get()


@contextmanager
def measure_active(stage):
    """Measure a nested stage when an outer timed call supplied a recorder."""

    recorder = current_recorder()
    if recorder is None:
        yield
        return
    with recorder.measure(stage):
        yield


def increment_active(name, amount=1):
    """Increment a counter on the active recorder, if one exists."""

    recorder = current_recorder()
    if recorder is not None:
        recorder.increment(name, amount)


def timed_active_stage(stage):
    """Decorate a hot helper with an optional nested stage timer."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with measure_active(stage):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def timed_stage(stage):
    """Measure a function when a ``host_timing`` keyword is supplied."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            recorder = kwargs.pop("host_timing", None)
            if recorder is None:
                return function(*args, **kwargs)
            token = _active_recorder.set(recorder)
            try:
                with recorder.measure(stage):
                    return function(*args, **kwargs)
            finally:
                _active_recorder.reset(token)

        return wrapped

    return decorate
