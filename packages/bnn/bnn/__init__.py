"""Fit and sample posteriors of Bayesian neural networks.

One configuration describes a run: data, model, prior and fit, and optionally
a sampler. ``bnn run`` brings it to completion
and continues it after any interruption; ``open_run`` reads its results, and
``run_of`` reads the finished run of a configuration file after checking that
the run matches it.
"""

import importlib
from typing import Any

_PUBLIC = {
    "RunConfig": "bnn.config",
    "load_configs": "bnn.config",
    "Run": "bnn.pipeline.run",
    "open_run": "bnn.pipeline.run",
    "run_of": "bnn.pipeline.run",
    "advance": "bnn.pipeline.pipeline",
    "Resources": "bnn.pipeline.workers",
}

__all__ = sorted(_PUBLIC)


def __getattr__(name: str) -> Any:
    module = _PUBLIC.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
