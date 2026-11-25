"""Are a sampler's kinetic dynamics worth learning?"""

from runner import Experiment

from . import budget, exact, frontier, predictive, summary

QUESTION = (
    "Do learned kinetic dynamics cross modes or estimate moments with fewer "
    "gradients than tuned classical ones, exactly or at a bias?"
)


def _measure(cell):
    return {
        "classical": frontier.measure_classical,
        "learned": frontier.measure_learned,
    }[cell.stage](cell)


EXPERIMENT = Experiment.join(
    QUESTION,
    {
        # Exact HMC under other momentum laws.
        "exact": Experiment(exact.design, exact.measure, summary.exact),
        # Unadjusted classical samplers and learned kinetic fields at one budget.
        "frontier": Experiment(frontier.design, _measure, summary.frontier),
        # The learned sampler across budgets, against a sampler told the axis.
        "budget": Experiment(budget.design, budget.measure, summary.budget),
        # Fields trained as published, on small regression networks.
        "predictive": Experiment(
            predictive.design, predictive.measure, summary.predictive
        ),
    },
)
