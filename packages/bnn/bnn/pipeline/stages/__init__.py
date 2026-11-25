"""The stages of a run.

A stage splits into tasks that are independent of each other, usually one per
chain group. Each stage has its own module; ``build_stages`` lists them in
execution order.
"""

from bnn.pipeline.stages.base import Controls, Stage, Task
from bnn.pipeline.stages.fit import FitStage
from bnn.pipeline.stages.sample import SampleStage

__all__ = ["Controls", "Stage", "Task", "build_stages"]


def build_stages() -> dict[str, Stage]:
    """All stages in execution order."""
    stages = (FitStage(), SampleStage())
    return {stage.name: stage for stage in stages}
