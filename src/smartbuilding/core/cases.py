"""ReferenceCase: a hand-made layout with the outcome the rule checker must produce for it."""

from typing import Self

from pydantic import Field, model_validator

from smartbuilding.core.base import Contract
from smartbuilding.core.primitives import CaseExpectation, Name, RelPath


class ReferenceCase(Contract):
    """Paths are relative to the folder that contains the case file."""

    case_id: Name
    description: str = Field(min_length=1)
    spec: RelPath
    layout: RelPath
    period: Name  # occupancy period from the occupant profile, for example 'typical'
    expect: CaseExpectation
    expected_violations: tuple[str, ...] = ()  # exact set of failing hard-check ids

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.expect == CaseExpectation.passes and self.expected_violations:
            raise ValueError("a case that passes must not list expected violations")
        if self.expect == CaseExpectation.fails and not self.expected_violations:
            raise ValueError("a case that fails must list the expected violations")
        if len(set(self.expected_violations)) != len(self.expected_violations):
            raise ValueError("expected_violations must not repeat")
        return self
