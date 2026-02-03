# FailGuard

Detect silent failures, drift, and stuck states in AI agents.

## The Problem

AI agents fail silently. They don't crash - they just slowly degrade:

- **Latency drift**: Response times creep up until timeouts
- **Stuck states**: Same output repeated endlessly
- **Cycles**: A→B→A→B patterns that never progress

Traditional error handling doesn't catch these. Your agent looks "fine" while burning tokens and failing users.

## The Solution

```python
from failguard import failguard

@failguard(max_latency_drift=2.0, max_identical_outputs=3)
def agent_step(query: str) -> str:
    return llm.complete(query)

# Raises FailGuardError if:
# - Latency exceeds 2x baseline
# - Same output repeated 3+ times
# - Cycle pattern detected (A→B→A→B)
```

## Installation

```bash
pip install failguard
```

## Features

- **Zero dependencies** - Only Python stdlib
- **Latency drift detection** - Catches gradual slowdowns
- **Stuck detection** - Identifies repeated identical outputs
- **Cycle detection** - Finds A→B→A→B patterns (complements LoopGuard)
- **Thread-safe** - Safe for concurrent use
- **Flexible API** - Decorator or inline Monitor class

## Usage

### Decorator API

```python
from failguard import failguard, FailGuardError

@failguard(
    max_latency_drift=3.0,      # Alert if latency > 3x baseline
    max_identical_outputs=5,    # Alert after 5 identical outputs
    stuck_window=60,            # Within 60 seconds
    detect_cycles=True,         # Detect A→B→A patterns
)
def agent_step(query: str) -> str:
    return llm.complete(query)

try:
    result = agent_step("What is 2+2?")
except FailGuardError as e:
    print(f"Failure detected: {e.failure_type}")
    print(f"Metrics: {e.metrics}")
```

### Custom Failure Handler

```python
def my_handler(status):
    logger.warning(f"Agent failing: {status.failure_types}")
    return "fallback response"

@failguard(max_identical_outputs=3, on_failure=my_handler)
def agent_step(query: str) -> str:
    return llm.complete(query)

# Returns "fallback response" instead of raising
```

### Inline Monitor

```python
from failguard import Monitor

monitor = Monitor(max_identical_outputs=3)

for step in workflow:
    result = agent.run(step)
    status = monitor.check(result, step_name=step)

    if status.is_stuck:
        print(f"Agent stuck: {status.identical_count} repeats")
        break
    if status.has_cycle:
        print(f"Cycle detected: {status.cycle_pattern}")
        break
    if status.has_latency_drift:
        print(f"Slowdown: {status.latency_drift_ratio}x baseline")
```

### With LoopGuard (Full Reliability Suite)

```python
from loopguard import loopguard
from failguard import failguard

@loopguard(max_repeats=5)        # Catch A→A→A (same args)
@failguard(detect_cycles=True)   # Catch A→B→A→B (different outputs)
def agent_action(query):
    return llm.complete(query)
```

## API Reference

### `@failguard(**options)`

Decorator for detecting failures.

| Option | Default | Description |
|--------|---------|-------------|
| `max_latency_drift` | 3.0 | Alert if latency > N × baseline |
| `max_identical_outputs` | 5 | Alert after N identical outputs |
| `stuck_window` | 60.0 | Time window (seconds) for stuck detection |
| `detect_cycles` | True | Detect repeating patterns |
| `cycle_min_length` | 2 | Minimum cycle length |
| `cycle_max_length` | 5 | Maximum cycle length |
| `on_failure` | None | Callback: `(FailureStatus) -> Any` |
| `raise_on_failure` | True | Raise FailGuardError on failure |

**Attached methods:**
- `func.reset()` - Clear all state
- `func.get_status()` - Get current status

### `Monitor(**options)`

Inline monitor with same options as decorator.

```python
monitor = Monitor()
status = monitor.check(value, step_name="step1", latency_ms=150)
monitor.reset()
```

### `FailureStatus`

Status object returned by checks.

| Field | Type | Description |
|-------|------|-------------|
| `has_failure` | bool | Any failure detected |
| `failure_types` | list | List of FailureType values |
| `has_latency_drift` | bool | Latency exceeded threshold |
| `latency_drift_ratio` | float | Current/baseline ratio |
| `is_stuck` | bool | Identical outputs exceeded threshold |
| `identical_count` | int | Number of identical outputs |
| `has_cycle` | bool | Cycle pattern detected |
| `cycle_pattern` | list | The repeating pattern |

### `FailGuardError`

Exception raised on failure.

```python
try:
    agent_step()
except FailGuardError as e:
    e.failure_type   # "stuck", "cycle", "latency_drift"
    e.message        # Human-readable description
    e.metrics        # Dict with relevant metrics
```

### `FailureType`

Constants for failure types:
- `FailureType.LATENCY_DRIFT`
- `FailureType.STUCK`
- `FailureType.CYCLE`

## Part of the Guard Suite

FailGuard is part of a reliability suite for AI agents:

- **[LoopGuard](https://pypi.org/project/loopguard/)** - Prevent infinite loops
- **[EvalGuard](https://pypi.org/project/evalguard/)** - Validate outputs
- **FailGuard** - Detect silent failures (this package)

## License

MIT
