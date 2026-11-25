"""Named components and the strict configuration base they share.

Every pluggable part of a run (model, fit, sampler) is a family: a registry of
names, each with its own configuration class. A family member is selected in
YAML by its ``name`` and validated against exactly the options that member
declares.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

T = TypeVar("T")


class Registry(Generic[T]):
    """Entries by name, with errors that list what is available."""

    def __init__(self, kind: str):
        """Name the kind of entry for error messages."""
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, entry: T) -> T:
        """Add an entry under a new name."""
        if name in self._entries:
            raise ValueError(f"Duplicate {self.kind} {name!r}.")
        self._entries[name] = entry
        return entry

    def __getitem__(self, name: str) -> T:
        """Look up an entry."""
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} {name!r}. Available: {', '.join(sorted(self))}."
            ) from None

    def __contains__(self, name: object) -> bool:
        """Whether a name is registered."""
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names."""
        return iter(self._entries)

    def items(self):
        """Registered names with their entries."""
        return self._entries.items()


class Config(BaseModel):
    """Immutable configuration that rejects unknown options."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, validate_default=True, use_enum_values=False
    )

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON- and YAML-ready values, including all defaults."""
        return self.model_dump(mode="json", serialize_as_any=True)


class Family(Config):
    """Base configuration of a family whose members are selected by ``name``.

    A field annotated with the family base accepts any registered member: the
    ``name`` of the given mapping selects the member's configuration class. The
    base of a family declares ``members = Registry("<kind>")``.
    """

    members: ClassVar[Registry[type[Family]]]
    name: str

    @classmethod
    def member(cls, name: str):
        """Class decorator registering a member configuration under ``name``."""

        def register(member: type[Family]) -> type[Family]:
            cls.members.register(name, member)
            return member

        return register

    @model_validator(mode="wrap")
    @classmethod
    def _select_member(cls, value: Any, handler):
        if isinstance(value, dict) and "members" in cls.__dict__:
            if "name" not in value:
                raise ValueError(
                    f"A {cls.members.kind} needs a 'name'. Available: "
                    + ", ".join(sorted(cls.members))
                )
            try:
                member = cls.members[value["name"]]
            except KeyError as error:
                raise ValueError(error.args[0]) from None
            return member.model_validate(value)
        return handler(value)
