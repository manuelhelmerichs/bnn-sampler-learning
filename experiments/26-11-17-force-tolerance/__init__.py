"""At which force error and relative cost does a gradient surrogate save compute?"""

from runner import Experiment

from .benchmark import design, measure
from .summary import summarize

EXPERIMENT = Experiment(design, measure, summarize)
