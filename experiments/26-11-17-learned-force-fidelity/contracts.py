"""Scientific contracts for frozen samplers, independent of model architecture.

These declarations check incompatible claims; they do not prove invariance of
an arbitrary implementation. Kernel tests must verify the declared proposal
accounting. Learning masks form a partial order, separate from validity.
Target access denotes exact evaluations during retained sampling. A learned
target surrogate may construct proposals while the final correction uses those
exact values. Learned acceptance rules and online adaptation need other balance
contracts and are outside the invariant mechanisms implemented here.
"""

from dataclasses import asdict, dataclass

COMPONENTS = (
    "initialization",
    "representation",
    "target",
    "force",
    "geometry",
    "integrator",
    "noise",
    "proposal",
    "schedule",
    "conditional",
    "correction",
)


@dataclass(frozen=True)
class SamplerContract:
    learned: tuple[str, ...]
    target_access: str
    proposal_accounting: str
    correction: str
    validity: str
    retention: str = "markov_chain"
    adaptation: str = "frozen"
    schedule: str = "fixed"
    selection_accounted: bool = False

    def __post_init__(self):
        if len(set(self.learned)) != len(self.learned) or not set(self.learned) <= set(
            COMPONENTS
        ):
            raise ValueError("Learned components must be distinct known components")
        choices = {
            "target_access": {"none", "value", "gradient", "value_and_gradient"},
            "proposal_accounting": {
                "reversible_volume_preserving",
                "evaluable_density",
                "exact_conditional",
                "implicit",
            },
            "correction": {"none", "metropolis"},
            "validity": {"finite_teacher", "approximate", "target_invariant"},
            "retention": {"markov_chain", "iid"},
            "adaptation": {"frozen", "online"},
            "schedule": {"fixed", "state_dependent"},
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Unknown sampler contract {name}")
        if self.correction == "metropolis":
            if self.target_access not in {
                "value",
                "value_and_gradient",
            } or self.proposal_accounting not in {
                "reversible_volume_preserving",
                "evaluable_density",
            }:
                raise ValueError(
                    "Metropolis needs target values and proposal accounting"
                )
            if self.retention != "markov_chain":
                raise ValueError("Metropolis retention includes repeated chain states")
        if self.validity == "target_invariant":
            if self.adaptation != "frozen":
                raise ValueError(
                    "Online adaptation needs a separate invariance argument"
                )
            if "correction" in self.learned:
                raise ValueError(
                    "A learned acceptance rule needs a separate balance contract"
                )
            if self.schedule == "state_dependent" and not self.selection_accounted:
                raise ValueError("State-dependent selection needs reverse accounting")
            if (
                self.correction != "metropolis"
                and self.proposal_accounting != "exact_conditional"
            ):
                raise ValueError(
                    "Target invariance needs correction or exact conditionals"
                )
            if (
                self.proposal_accounting == "exact_conditional"
                and "conditional" in self.learned
            ):
                raise ValueError(
                    "An approximate conditional is not an exact conditional"
                )

    def to_dict(self):
        result = asdict(self)
        result["learned"] = list(self.learned)
        return result
