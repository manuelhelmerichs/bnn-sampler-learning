"""Workspace-wide test configuration."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent

# Set limits before collection imports numerical libraries. Explicit caller
# settings (including extra CPU devices used by pipeline tests) take precedence.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def pytest_collection_modifyitems(config, items):
    """Reject accidental imports of a different test file with the same name."""
    for item in items:
        if (
            isinstance(item, pytest.Function)
            and Path(item.module.__file__).resolve() != item.path.resolve()
        ):
            raise pytest.UsageError(
                f"{item.nodeid} imported a different test file: {item.module.__file__}"
            )


@pytest.fixture(autouse=True)
def _restore_jax_precision():
    """A test that enables float64 must not change the dtypes seen by later tests."""
    # Architecture and CI checks run with only the dev dependency group installed.
    jax = sys.modules.get("jax")
    if jax is None:
        yield
        return
    enabled = bool(jax.config.jax_enable_x64)
    yield
    jax.config.update("jax_enable_x64", enabled)
