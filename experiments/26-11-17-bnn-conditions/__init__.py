"""Where do BNN posteriors sit on the axes that decide whether a learned component pays?"""

from runner import Experiment

from .conditions import design, measure
from .summary import summarize

EXPERIMENT = Experiment(design, measure, summarize)
