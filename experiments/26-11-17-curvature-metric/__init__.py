"""Does a constant curvature metric make a sampler of a BNN posterior faster?"""

from runner import Experiment

from .benchmark import design, measure
from .summary import summarize

EXPERIMENT = Experiment(design, measure, summarize)
